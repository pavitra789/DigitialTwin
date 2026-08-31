"""
decision_engine.py
------------------
Core TraCI loop for the Digital Twin Smart Traffic Management System.

This is the module every other feature (AI prediction, route recommendation,
FastAPI/WebSocket layer) plugs into. It:

  1. Connects to a running SUMO simulation via TraCI.
  2. Reads live per-lane traffic data every simulation step (the "Digital
     Twin" state).
  3. Runs a simple decision engine each cycle:
       - Emergency vehicle priority (checked first, overrides everything)
       - Adaptive traffic signal control (density-based green time)
  4. Pushes the decision back into SUMO via TraCI.

Requirements:
    pip install traci sumolib

Before running:
    - Create/obtain a SUMO network (.net.xml) and route file (.rou.xml).
    - Create a .sumocfg file referencing them.
    - Update SUMO_CONFIG_PATH below.
    - Update TRAFFIC_LIGHT_ID and LANE_IDS to match your network.

Run:
    python decision_engine.py
"""

import os
import sys
import shutil

# Ensure SUMO_HOME is configured
if "SUMO_HOME" not in os.environ:
    default_sumo_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
    ]
    for path in default_sumo_paths:
        if os.path.exists(path):
            os.environ["SUMO_HOME"] = path
            os.environ["PATH"] += os.pathsep + os.path.join(path, "bin")
            break

import traci
import sumolib

# ---------------------------------------------------------------------------
# CONFIG — update these to match your SUMO network
# ---------------------------------------------------------------------------
SUMO_BINARY = "sumo"              # use "sumo" for console simulation, "sumo-gui" for visual UI
SUMO_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "network", "simulation.sumocfg")

TRAFFIC_LIGHT_ID = "TL1"          # ID of the traffic light junction to control
LANE_IDS = ["north_in_0", "east_in_0", "south_in_0", "west_in_0"]

MIN_GREEN_TIME = 15               # seconds
MAX_GREEN_TIME = 60               # seconds
DEFAULT_GREEN_TIME = 30           # seconds when no strong signal either way

EMERGENCY_VEHICLE_TYPE = "emergency"   # vType id used for ambulances in .rou.xml

# Phase indices in your traffic light's phase program (from intersection.net.xml)
# Phase 0 = North-South green, Phase 2 = East-West green
PHASE_FOR_LANE_DIRECTION = {
    "north_in_0": 0,
    "south_in_0": 0,
    "east_in_0": 2,
    "west_in_0": 2,
}


# ---------------------------------------------------------------------------
# DIGITAL TWIN STATE — mirrors the physical/simulated intersection
# ---------------------------------------------------------------------------
class LaneTwin:
    """Virtual representation of one lane's current state."""

    def __init__(self, lane_id):
        self.lane_id = lane_id
        self.vehicle_count = 0
        self.avg_speed = 0.0
        self.queue_length = 0
        self.waiting_time = 0.0
        self.has_emergency_vehicle = False

    def update(self):
        vehicle_ids = traci.lane.getLastStepVehicleIDs(self.lane_id)
        self.vehicle_count = len(vehicle_ids)
        self.avg_speed = traci.lane.getLastStepMeanSpeed(self.lane_id)
        self.queue_length = traci.lane.getLastStepHaltingNumber(self.lane_id)
        self.waiting_time = traci.lane.getWaitingTime(self.lane_id)

        self.has_emergency_vehicle = any(
            traci.vehicle.getTypeID(vid) == EMERGENCY_VEHICLE_TYPE
            for vid in vehicle_ids
        )

    def __repr__(self):
        emergency_str = " [EMERGENCY VEHICLE]" if self.has_emergency_vehicle else ""
        return (
            f"  LaneTwin({self.lane_id:<12}: vehicles={self.vehicle_count:>2}, "
            f"speed={self.avg_speed:>4.1f} m/s, queue={self.queue_length:>2}, "
            f"waiting_time={self.waiting_time:>4.1f}s{emergency_str})"
        )


class DigitalTwin:
    """Holds the current virtual state of all monitored lanes."""

    def __init__(self, lane_ids):
        self.lanes = {lid: LaneTwin(lid) for lid in lane_ids}

    def sync(self):
        """Pull the latest state from SUMO for every lane (call every step)."""
        for lane in self.lanes.values():
            lane.update()

    def get_state_snapshot(self):
        """Return a plain dict — useful for logging / API / AI features."""
        return {
            lid: {
                "vehicle_count": lt.vehicle_count,
                "avg_speed": lt.avg_speed,
                "queue_length": lt.queue_length,
                "waiting_time": lt.waiting_time,
                "has_emergency_vehicle": lt.has_emergency_vehicle,
            }
            for lid, lt in self.lanes.items()
        }


# ---------------------------------------------------------------------------
# DECISION ENGINE
# ---------------------------------------------------------------------------
def check_emergency_priority(twin: DigitalTwin):
    """
    Returns the lane_id of a lane holding an emergency vehicle, or None.
    This check should run BEFORE adaptive signal logic — it overrides it.
    """
    for lane_id, lane in twin.lanes.items():
        if lane.has_emergency_vehicle:
            return lane_id
    return None


def compute_adaptive_green_time(twin: DigitalTwin, lane_id: str) -> int:
    """
    Simple density-based green time formula.
    Denser lane -> more green time, bounded by MIN/MAX.
    """
    lane = twin.lanes[lane_id]
    # crude scaling: every 5 extra vehicles beyond baseline adds ~5 seconds
    extra_time = (lane.vehicle_count // 5) * 5
    green_time = DEFAULT_GREEN_TIME + extra_time
    return max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, green_time))


def apply_signal_decision(lane_id: str, green_time: int):
    """Pushes a phase + duration decision back into SUMO via TraCI."""
    phase = PHASE_FOR_LANE_DIRECTION.get(lane_id)
    if phase is None:
        print(f"[WARN] No phase mapping for lane {lane_id}, skipping.")
        return
    traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, phase)
    traci.trafficlight.setPhaseDuration(TRAFFIC_LIGHT_ID, green_time)
    direction_name = "North-South Green" if phase == 0 else "East-West Green"
    print(f"  [DECISION] Lane '{lane_id}' -> Phase {phase} ({direction_name}), Green Duration: {green_time}s")


def run_decision_cycle(twin: DigitalTwin):
    """One full decision cycle: emergency check first, then adaptive control."""
    emergency_lane = check_emergency_priority(twin)

    if emergency_lane:
        print(f"  [EMERGENCY OVERRIDE] Ambulance detected on approach '{emergency_lane}' -> Granting Priority Green Wave!")
        apply_signal_decision(emergency_lane, MAX_GREEN_TIME)
        return

    # No emergency: pick the densest lane and give it adaptive green time
    busiest_lane = max(
        twin.lanes.values(), key=lambda lt: lt.vehicle_count
    ).lane_id
    green_time = compute_adaptive_green_time(twin, busiest_lane)
    apply_signal_decision(busiest_lane, green_time)


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
def main():
    # Detect available binary (sumo or sumo-gui if user specifies --gui)
    use_gui = "--gui" in sys.argv
    binary = "sumo-gui" if use_gui else SUMO_BINARY

    # Fallback to full path if not on PATH
    if shutil.which(binary) is None and "SUMO_HOME" in os.environ:
        binary_full = os.path.join(os.environ["SUMO_HOME"], "bin", f"{binary}.exe")
        if os.path.exists(binary_full):
            binary = binary_full

    sumo_cmd = [binary, "-c", SUMO_CONFIG_PATH, "--no-warnings", "true"]
    print(f"[START] Launching SUMO Simulation via TraCI: {sumo_cmd[0]}")
    traci.start(sumo_cmd)

    twin = DigitalTwin(LANE_IDS)

    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step < 200:
            traci.simulationStep()
            twin.sync()

            # Run the decision engine every 10 simulation steps (tune as needed)
            if step % 10 == 0:
                print(f"\n==================== Simulation Step {step} ====================")
                for lane in twin.lanes.values():
                    print(lane)
                run_decision_cycle(twin)

            step += 1
    finally:
        traci.close()
        print(f"\n[FINISHED] SUMO Simulation finished successfully after {step} steps.")


if __name__ == "__main__":
    main()
