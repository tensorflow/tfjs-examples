// Calculate the arithmetic mean of a vector.
//
// Args:
//   vector: The vector represented as an Array of Numbers.
//
// Returns:
//   The arithmetic mean.
const mean = (vector) => {
  let sum = 0;
  for (const x of vector) {
    sum += x;
  }
  return sum / vector.length;
};

// Calculate the standard deviation of a vector.
//
// Args:
//   vector: The vector represented as an Array of Numbers.
//
// Returns:
//   The standard deviation.
const stddev = (vector) => {
  let squareSum = 0;
  const vectorMean = mean(vector);
  for (const x of vector) {
    squareSum += (x - vectorMean) * (x - vectorMean);
  }
  return Math.sqrt(squareSum / (vector.length - 1));
};

// Normalize a vector by its mean and standard deviation.
const normalizeVector = (vector, vectorMean, vectorStddev) => {
  return vector.map(x => (x - vectorMean) / vectorStddev);
};

module.exports = {
  mean,
  stddev,
  normalizeVector
}
