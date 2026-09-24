# Time-indexed human-occupancy risk map on the MAPF grid (same cell indexing as the
# mapf_<map>.yaml file used by remroc_mapf_solver: x in [0, width), y in [0, height)).
#
# data[(t * height + y) * width + x] is the predicted probability in [0, 1] that a
# human occupies cell (x, y) at MAPF time step t, i.e. at time
# header.stamp + t * step_duration. Layers t >= horizon use the last layer, which a
# coordinator should set to a static (time-averaged) prior.
std_msgs/Header header
float64 step_duration
uint32 width
uint32 height
uint32 horizon
float32[] data
