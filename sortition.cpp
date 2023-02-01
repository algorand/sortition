#include "sortition.h"
#include <boost/math/distributions/binomial.hpp>
#if !defined(BOOST_VERSION)
#error Boost version not defined
#endif
#if BOOST_VERSION != 106501
#error Boost version does not match 1.65.1
#endif

uint64_t sortition_binomial_cdf_walk(double n, double p, double ratio, uint64_t money) {
  boost::math::binomial_distribution<double> dist(n, p);
  for (uint64_t j = 0; j < money; j++) {
    // Get the cdf
    double boundary = cdf(dist, j);

    // Found the correct boundary, break
    if (ratio <= boundary) {
      return j;
    }
  }
  return money;
}
