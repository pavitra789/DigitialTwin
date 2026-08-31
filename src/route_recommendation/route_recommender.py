"""
route_recommender.py
--------------------
Static Route Recommendation Engine.
Monitors predicted congestion on primary intersection corridors. When an approach
is predicted to experience HIGH or CRITICAL congestion, incoming vehicles on that
route are switched to precomputed static alternate paths to balance network load.
"""

from typing import Dict, List, Optional, Tuple
import traci
from ..digital_twin.digital_twin_system import DigitalTwin


# Precomputed static alternate routes for OD pairs
# Key: Primary Route ID -> Value: Alternate Route ID
STATIC_ALTERNATE_ROUTES = {
    "r_NS": "r_NS_alt1",  # Divert from North-South to North-West
    "r_SN": "r_SN_alt1",  # Divert from South-North to South-East
    "r_EW": "r_EW_alt1",  # Divert from East-West to East-North
    "r_WE": "r_WE_alt1",  # Divert from West-East to West-South
}

# Mapping lanes to primary routes passing through them
LANE_TO_PRIMARY_ROUTE = {
    "north_in_0": "r_NS",
    "south_in_0": "r_SN",
    "east_in_0": "r_EW",
    "west_in_0": "r_WE",
}


class StaticRouteRecommender:
    """
    Evaluates predicted congestion and applies static alternate routes to approaching vehicles.
    """

    def __init__(
        self,
        congestion_threshold: str = "HIGH",
        alternate_routes: Optional[Dict[str, str]] = None
    ):
        self.congestion_threshold = congestion_threshold
        self.alternate_routes = alternate_routes or STATIC_ALTERNATE_ROUTES
        self.rerouted_vehicles_history: List[dict] = []
        self.reroute_count: int = 0

    def should_reroute(self, congestion_level: str) -> bool:
        """Determines if the predicted congestion level warrants alternate route diversion."""
        severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        target_rank = severity_rank.get(self.congestion_threshold, 2)
        current_rank = severity_rank.get(congestion_level, 0)
        return current_rank >= target_rank

    def evaluate_and_reroute(
        self,
        twin: DigitalTwin,
        lane_predictions: Dict[str, Tuple[float, str]]
    ) -> List[dict]:
        """
        Scans all lanes. For lanes with high predicted congestion, finds active approaching
        vehicles on the primary route and reroutes them to the alternate route via TraCI.

        Args:
            twin: DigitalTwin state
            lane_predictions: Dict mapping lane_id -> (predicted_count, congestion_level)

        Returns:
            List of reroute actions taken during this step.
        """
        actions = []

        for lane_id, (pred_count, level) in lane_predictions.items():
            if not self.should_reroute(level):
                continue

            primary_route = LANE_TO_PRIMARY_ROUTE.get(lane_id)
            if not primary_route or primary_route not in self.alternate_routes:
                continue

            alt_route = self.alternate_routes[primary_route]
            lane_obj = twin.lanes.get(lane_id)
            if not lane_obj:
                continue

            # Select approaching passenger vehicles in this lane
            for vid in lane_obj.active_vehicle_ids:
                # Do not reroute emergency vehicles
                veh_twin = twin.vehicles.get(vid)
                if veh_twin and veh_twin.is_emergency:
                    continue

                # Check if vehicle has not already been rerouted
                if any(r["vehicle_id"] == vid for r in self.rerouted_vehicles_history[-50:]):
                    continue

                # Attempt TraCI route switch
                if traci.isLoaded():
                    try:
                        curr_route = traci.vehicle.getRouteID(vid)
                        if curr_route == primary_route:
                            traci.vehicle.setRouteID(vid, alt_route)
                            action_record = {
                                "step": twin.step_count,
                                "vehicle_id": vid,
                                "lane_id": lane_id,
                                "original_route": primary_route,
                                "recommended_route": alt_route,
                                "reason": f"Predicted {level} congestion ({pred_count} vehs)",
                            }
                            self.rerouted_vehicles_history.append(action_record)
                            self.reroute_count += 1
                            actions.append(action_record)
                            print(f"[SMART ROUTING] Rerouted {vid} from {primary_route} -> {alt_route} (Reason: {level} on {lane_id})")
                    except Exception as err:
                        print(f"[WARN] Failed to reroute vehicle {vid}: {err}")
                else:
                    # Offline / simulated action
                    action_record = {
                        "step": twin.step_count,
                        "vehicle_id": vid,
                        "lane_id": lane_id,
                        "original_route": primary_route,
                        "recommended_route": alt_route,
                        "reason": f"Predicted {level} congestion ({pred_count} vehs)",
                    }
                    self.rerouted_vehicles_history.append(action_record)
                    self.reroute_count += 1
                    actions.append(action_record)

        return actions

    def get_summary(self) -> dict:
        """Returns routing telemetry summary for Digital Twin snapshots."""
        return {
            "total_rerouted": self.reroute_count,
            "recent_actions": self.rerouted_vehicles_history[-10:],
        }
