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
  // Binary search for find the lowest stake <= cdf
  uint64_t low = 0;
  uint64_t high = money - 1;
  while (low + 1 < high) {
    const uint64_t mid = low + (high - low) / 2;

    // Get the cdf
    if (ratio <= cdf(dist, mid)) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  // Found the correct boundary
  if (ratio <= cdf(dist, low)) {
    return low;
  }
  if (ratio <= cdf(dist, high)) {
    return high;
  }
  return money;
}
