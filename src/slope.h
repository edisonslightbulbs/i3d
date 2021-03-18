#ifndef SLOPE_H
#define SLOPE_H

#include <vector>
#include <cmath>
#include <algorithm>

namespace slope {

    float largest(std::vector<float> x){
    std::sort(x.begin(), x.end(), std::greater<float>());

        float gradient = 0;
        for(int i = 1; i < x.size() - 1; i++){
            /** evaluate slope from current value to the next */
            std::cout << x[i] << " ---- " << x[i + 1] << std::endl;
            // float vecGradient = x[i + 1] - x [i];
            // if (vecGradient > gradient){
            //     gradient = x[i];
            // }
        }
        return gradient;
    }

    // float smallest(std::vector<float> x){
    //     float gradient = __DBL_MAX__;
    //     for(int i = 0; i < x.size(); i++){
    //         /** evaluate slope from current value to the next */
    //         float vecGradient = x[i + 1] - x [i];
    //         if (vecGradient < gradient){
    //             gradient = x[i];
    //         }
    //     }
    //     return gradient;
    // }
}
#endif /* SLOPE_H */