import TensorFlow

/// K-neighbors classifier.
///
/// Classifier implementing the k-nearest neighbors vote.
public class KNeighborsClassifier {
    /// The order of the norm of the difference: `||a - b||_p`.
    var p: Int
    /// Weight function used in prediction.
    var weights: String
    /// Number of neighbors.
    var neighborCount: Int
    /// The training data.
    var data: Tensor<Float>
    /// The target class correspoing to training data.
    var labels: Tensor<Int32>
  
    /// Create a K neighbors classifier model.
    ///
    /// - Parameters
    ///   - neighborCount: Number of neighbors to use, default to `5`.
    ///   - weights: Weight function used in prediction. Possible values `uniform` - uniform
    ///     weighted, `distance` - weight point by inverse of their distance. Default set to
    ///     `distance`.
    ///   - p: The order of the norm of the difference: `||a - b||_p`, default set to `2`.
    public init(
        neighborCount: Int = 5,
        weights: String = "distance",
        p: Int = 2
    ) {
        precondition(neighborCount >= 1, "Neighbor count must be greater than or equal to one.")
        precondition(weights == "uniform" || weights == "distance",
            "Weights must be either 'uniform' or 'distance'.")
        precondition(p > 1, "p must be positive.")

        self.neighborCount = neighborCount
        self.weights = weights
        self.p = p
        self.data = Tensor<Float>([0.0])
        self.labels = Tensor<Int32>([0])
    }
  
    /// Fit a K-neighbors classifier model.
    ///
    /// - Parameters
    ///   - data: Training data with shape `[sample count, feature count]`.
    ///   - labels: Target value with shape `[sample count]`.
    public func fit(data: Tensor<Float>, labels: Tensor<Int32>) {
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")
        precondition(data.shape[1] >= 1,
            "Data must have feature count greater than or equal to one.")
        precondition(labels.shape[0] > 0, "Labels must have a positive sample count.")
        precondition(data.shape[0] == labels.shape[0],
            "Data and labels must have the same sample count.")

        self.data = data
        self.labels = labels
    }

    /// Returns the weights of each neighbor.
    ///
    /// - Parameters
    ///   - distances: Contains the distance between test data and top neighbors.
    ///   - labels: Contains the classes of neighbors.
    /// - Returns - The weights of each neighbors.
    internal func computeWeights(
        distances: Tensor<Float>,
        labels: Tensor<Float>
    ) -> Tensor<Float> {
        var labelsAndWeightsTensor = Tensor<Float>(zeros: [distances.shape[0], 2])
        
        // In `uniform` weighing method each class have same weight.
        // In `distance` weighing method weight vary based on the inverse of distance.
        if self.weights == "uniform" {
            for i in 0..<distances.shape[0] {
                labelsAndWeightsTensor[i][0] = labels[i]
                labelsAndWeightsTensor[i][1] = Tensor<Float>(1.0)
            }
            return labelsAndWeightsTensor
        } else if self.weights == "distance" {
            var matched = Tensor<Float>(zeros: [1, 2])
            for i in 0..<distances.shape[0] {
                if distances[i] == Tensor<Float>([0.0]) {
                    matched[0][0] = labels[i]
                    matched[0][1] = Tensor<Float>(1.0)
                    return matched
                }
                labelsAndWeightsTensor[i][0] = labels[i]
                labelsAndWeightsTensor[i][1] = Tensor<Float>(1.0 / distances[i])
            }
            return labelsAndWeightsTensor
        }
        return Tensor<Float>(0.0)
    } 

    /// Returns the predicted classification.
    ///
    /// - Parameter test: Input data to be classified.
    /// - Returns: Predicted classification.
    internal func predictSingleSample(_ test: Tensor<Float>) -> Tensor<Int32> {
        var distances = Tensor<Float>(zeros: [self.data.shape[0]])
        var maxLabel = Tensor<Int32>(zeros: [self.neighborCount])
        var maxDistances: Tensor<Float>
        var maxIndex: Tensor<Int32>
        var classes: Tensor<Int32>
        var indices: Tensor<Int32>
      
        // Calculate the distance between test and all data points.
        for i in 0..<self.data.shape[0] {
            distances[i] = minkowskiDistance(self.data[i], test, p: self.p)
        }

        // Find the top neighbor with minimum distance.
        (maxDistances, maxIndex) =
            Raw.topKV2(distances, k: Tensor<Int32>(Int32(data.shape[0])), sorted: true)
        maxDistances = Raw.reverse(maxDistances, dims: Tensor<Bool>([true]))
        maxDistances = maxDistances
            .slice(lowerBounds: Tensor<Int32>([0]),
                sizes: Tensor<Int32>([Int32(self.neighborCount)]))

        maxIndex = Raw.reverse(maxIndex, dims: Tensor<Bool>([true]))
        maxIndex = maxIndex
            .slice(lowerBounds: Tensor<Int32>([0]),
                sizes: Tensor<Int32>([Int32(self.neighborCount)]))

        for i in 0..<self.neighborCount {
            maxLabel[i] = self.labels[Int(maxIndex[i].scalarized())]
        }

        // Weights the neighbors based on their weighing method.
        let labelsAndWeightsTensor = computeWeights(
            distances: maxDistances, labels: Tensor<Float>(maxLabel))

        (classes, indices) = Raw.unique(Tensor<Int32>(maxLabel))
        
        var kClasses = Tensor<Int32>(zeros: [classes.shape[0]])
        var kWeights = Tensor<Float>(zeros: [classes.shape[0]])

        // Add weights based on their neighbors class.
        for i in 0..<labelsAndWeightsTensor.shape[0] {
            for j in 0..<classes.shape[0] {
                if labelsAndWeightsTensor[i][0] == Tensor<Float>(classes[j]) {
                    kClasses[j] = Tensor<Int32>(classes[j])
                    kWeights[j] = kWeights[j] + labelsAndWeightsTensor[i][1]
                }
            }
        }
        
        // Returns class with highest weight.
        let resultSet = Raw.topKV2(kWeights, k: Tensor<Int32>(1), sorted: true)
        let classIndex = Int(resultSet.indices[0].scalarized())
        return kClasses[classIndex]
    }

    /// Returns classification.
    ///
    /// - Parameter data: Prediction data with shape `[sample count, feature count]`.
    /// - Returns: Classification for test data.  
    public func prediction(for data: Tensor<Float>) -> Tensor<Int32> {
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")

        var predictions = Tensor<Int32>(zeros: [data.shape[0]])
        for i in 0..<data.shape[0] {
            predictions[i] = predictSingleSample(data[i])
        }
        return predictions
    }
    
    /// Returns the mean accuracy.
    ///
    /// - Parameters
    ///   - data: Sample data with shape `[sample count, feature count]`.
    ///   - labels: Target label with shape `[sample count]`.
    /// - Returns: Returns the mean accuracy on the given test data and labels.
    public func score(data: Tensor<Float>, labels: Tensor<Int32>) -> Float {
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")
        precondition(labels.shape[0] > 0, "Labels must have a positive sample count.")
        precondition(data.shape[0] == labels.shape[0],
            "Data and labels must have the same sample count.")

        let predictedLabels = self.prediction(for: data)
        var count: Int = 0
        for i in 0..<data.shape[0] {
            if predictedLabels[i] == labels[i] {
                count = count + 1
            }
        }
        let score = Float(count) / Float(labels.shape[0])
        return score
    }
}