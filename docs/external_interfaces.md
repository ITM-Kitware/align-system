# External Interfaces

The ALIGN System can interface with a few difference services provided
by other performers.  These interfaces may require additional setup
assuming you need to run the services locally for testing / debugging.

Once the external interfaces have been set up, you can have an ADM
talk to the TA3 service by overriding the `/interface` config option,
setting it to `ta3`.

## TA3 Action-based API

The code for the TA3 Action-based service can be found at: [TA3 Evaluation Server API
Repository](https://github.com/NextCenturyCorporation/itm-evaluation-server).

There's a corresponding client module: [TA3 Evaluation Client](https://github.com/NextCenturyCorporation/itm-evaluation-client)

## Soartech's TA1 API (must request access)

Soartech's TA1 service code can be found at: [Soartech's TA1
API](https://github.com/ITM-Soartech/ta1-server-mvp).  This API
provides alignment scores for answered probes and scenarios.

## ADEPT's TA1 API

ADEPT's TA1 service code can be found at: [ADEPT's TA1
API](https://gitlab.com/itm-ta1-adept-shared/adept_server).
This API provides alignment scores for answered probes and scenarios.
