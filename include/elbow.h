#ifndef ELBOW_H
#define ELBOW_H

#include <vector>

namespace elbow {

float secondDerivative(std::vector<float>& x);
float firstDerivative(const std::vector<float>& x);
float analyze(const std::vector<float>& x);

}
#endif /* ELBOW_H */
