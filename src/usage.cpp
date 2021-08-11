#include "usage.h"
#include "logger.h"

void usage::prompt(const int& code)
{
    switch (code) {
    case (ABOUT):
        LOG(INFO) << "-- 3DINTACT is currently unstable and should only be "
                     "used for academic purposes!";
        LOG(INFO) << "-- press ESC to exit";
        break;
    default:
        break;
    }
}
