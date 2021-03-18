#ifndef ELBOW_H
#define ELBOW_H

#include <vector>
#include <cmath>
#include <map>
#include <algorithm>

namespace elbow {

    /** evaluate the second derivative */
    float secondDerivative(std::vector<float>& x)
    {
        /** sort in descending order */
        std::sort(x.begin(), x.end(), std::greater<>());

        /** secant-line hyper-parameter */
        const int SIZE_OF_SECANT_LINE = (int)x.size() / 100;

        /** find max rate of change  */
        float yVal = 0;
        float slope = 0;
        for (int i = SIZE_OF_SECANT_LINE; i < x.size(); i++) {
            float roc = std::abs(x[i] - x[i - SIZE_OF_SECANT_LINE]);
            if(roc > slope){
                yVal = x[i];
                slope = roc;
            }
        }
        const std::string xMax = io::pwd() + "/build/bin/xMax.csv";
        io::write_val(x, xMax);
        // const std::string xMin = io::pwd() + "/build/bin/xMin.csv";
        // io::write_val(x, xMin);

        // const std::string yMax = io::pwd() + "/build/bin/yMax.csv";
        // io::write_val(x, yMax);

        // const std::string yMin = io::pwd() + "/build/bin/yMin.csv";
        // io::write_val(x, yMin);
        return yVal;
    }

    /** evaluate the first derivative */
    float firstDerivative(const std::vector<float>& x){

        /** keep track of the axis value (yValue) and its
         *  corresponding derivative d1 { yVal, d1 } using a map */
        std::map<float, float> derivativeMap;
        std::map<float, float>::iterator it;

        /** evaluate the first derivative */
        float yVal = 0;
        std::vector<float> derivatives;
        for(int i = 0; i < x.size() - 1; i++){
            float d1 = x[i + 1] - x[i];
            yVal = x[i];
            derivativeMap.insert(std::pair<float, float>(d1, yVal ));
            derivatives.push_back(d1);
        }

        /** estimate elbow point using first order derivatives */
        float elbow = secondDerivative(derivatives);
        it = derivativeMap.find(elbow);

        std::cout << "  max value: " << *std::max_element(x.begin(), x.end()) << std::endl;
        std::cout << "  min value: " << *std::min_element(x.begin(), x.end()) << std::endl;
        std::cout << "elbow value: " << it->second << std::endl;

        return it->second;
    }

    /** analyze successive axis values*/
    float analyzeAxis(const std::vector<float>& x){
        float axisEdge = firstDerivative(x);
        return axisEdge;
    }
}
#endif /* ELBOW_H */


