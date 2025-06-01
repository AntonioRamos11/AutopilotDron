# PX4 Parameters for High-Speed Accurate Navigation

This document outlines the key PX4 parameters that need to be modified for high-speed, accurate point-to-point navigation.

## Position Controller Parameters

| Parameter | Recommended Value | Description |
|-----------|-----------------|-------------|
| `MPC_XY_VEL_MAX` | 15.0 | Maximum horizontal velocity (m/s) |
| `MPC_Z_VEL_MAX_UP` | 5.0 | Maximum ascent velocity (m/s) |
| `MPC_Z_VEL_MAX_DN` | 3.0 | Maximum descent velocity (m/s) |
| `MPC_ACC_HOR_MAX` | 8.0 | Maximum horizontal acceleration (m/s²) |
| `MPC_JERK_MAX` | 20.0 | Maximum jerk (m/s³) |
| `MPC_XY_P` | 1.0 | Position proportional gain |
| `MPC_XY_VEL_P_ACC` | 3.0 | Velocity proportional gain |
| `MPC_XY_VEL_I_ACC` | 1.5 | Velocity integral gain |
| `MPC_XY_VEL_D_ACC` | 0.2 | Velocity derivative gain |

## EKF2 Parameters (Sensor Fusion)

| Parameter | Recommended Value | Description |
|-----------|-----------------|-------------|
| `EKF2_GPS_CTRL` | 1 | Enable GPS control |
| `EKF2_HGT_MODE` | 3 | Height sensor source (3 = GPS) |
| `EKF2_GPS_DELAY` | 0 | GPS measurement delay (ms) |
| `EKF2_GPS_POS_X` | Measure | GPS X position in body frame (m) |
| `EKF2_GPS_POS_Y` | Measure | GPS Y position in body frame (m) |
| `EKF2_GPS_POS_Z` | Measure | GPS Z position in body frame (m) |
| `EKF2_MAG_TYPE` | 1 | Magnetometer fusion type |
| `EKF2_GPS_CHECK` | 247 | GPS quality checks (247 = all checks enabled) |

## RTK GPS Parameters

| Parameter | Recommended Value | Description |
|-----------|-----------------|-------------|
| `GPS_UBX_DYNMODEL` | 7 | u-blox dynamic model (7 = airborne <2g) |
| `GPS_UBX_MODE` | 4 | u-blox device mode (4 = RTK fixed) |
| `GPS_YAW_OFFSET` | Measure | Antenna array heading offset (degrees) |
| `GPS_PITCH_OFFSET` | Measure | Antenna array pitch offset (degrees) |

## Mission Parameters

| Parameter | Recommended Value | Description |
|-----------|-----------------|-------------|
| `NAV_ACC_RAD` | 2.0 | Acceptance radius for waypoints (m) |
| `NAV_FT_DST` | 3.0 | Distance to first track point in missions (m) |
| `NAV_MIN_FT_HT` | 5.0 | Minimum flight height during mission (m) |
| `MIS_YAW_TMT` | 0 | Time constant for yaw maneuvers in missions |
| `MIS_YAW_ERR` | 12.0 | Max yaw error in degrees during mission |

## Safety Parameters

| Parameter | Recommended Value | Description |
|-----------|-----------------|-------------|
| `NAV_RCL_ACT` | 2 | RC Loss failsafe mode (2 = Return to Land) |
| `COM_LOW_BAT_ACT` | 3 | Low battery failsafe (3 = Return to Land) |
| `COM_DISARM_LAND` | 2 | Auto-disarm time after landing (seconds) |
| `GF_ACTION` | 1 | Geofence violation action (1 = Warning) |

## Installation Instructions

1. Connect to the vehicle via QGroundControl
2. Go to the Vehicle Setup -> Parameters section
3. Search for each parameter by name and update to the recommended value
4. Click "Save" after changing each parameter
5. Reboot the flight controller to apply all changes

**Note**: These parameters should be tuned incrementally in a safe test environment. Start with lower speed/acceleration values and gradually increase as performance is validated.