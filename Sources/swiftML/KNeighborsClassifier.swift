import TensorFlow

/// K neighbors classifier.
///
/// Classifier implementing the k-nearest neighbors vote.
public class KNeighborsClassifier {

    var p: Int
    var weights: String
    var data: Tensor<Float>
    var labels: Tensor<Float>
    var neighbors: Int
  
    /// Create a K neighbors classifier model.
    ///
    /// - Parameters
    ///   - neighbors: Number of neighbors to use, default to 5.
    ///   - weights: Weight function used in prediction. Possible values 'uniform' - uniform
    ///     weighted, 'distance' - weight point by inverse of thier distance. default set to
    ///     'distance'.
    ///   - p: The distance metric to use for the tree, default set to 2.
    public init(
        neighbors: Int = 5,
        weights: String = "distance",
        p: Int = 2
    ) {
        precondition(neighbors > 1, "Neighbors must be greater than one.")
        precondition(weights == "uniform" || weights == "distance",
            "weight must be either 'uniform' or 'distance'.")
        precondition(p > 1, "p must be greater than one.")

        self.neighbors = neighbors
        self.weights = weights
        self.p = p
        self.data = Tensor<Float>([0.0])
        self.labels = Tensor<Float>([0.0])
    }
  
    /// fit Kneighbors classifier model.
    ///
    /// - Parameters
    ///   - data: Training data Tensor<Float> of shape [number of samples, number of features].
    ///   - labels: Target value Tensor<Float> of shape [number of samples].
    public func fit(data: Tensor<Float>, labels: Tensor<Float>) {

        precondition(data.shape[0] == labels.shape[0],
            "data and labels must have same number of samples.")
        precondition(data.shape[0] > 0, "data must be non-empty.")
        precondition(data.shape[1] >= 1, "data must have atleast single feature.")
        precondition(labels.shape[0] > 0, "labels must be non-empty.")

        self.data = data
        self.labels = labels
    }

    /// Return the weights of each neighbors.
    ///
    /// - Parameters
    ///   - distances: Tensor contains the distance between test tensor and top neighbors.
    ///   - labels: Tensor contains the classes of neighbors.
    internal func computeWeights(
        distances: Tensor<Float>,
        labels: Tensor<Float>
    ) -> Tensor<Float> {
  
        var labelsAndWeightsTensor = Tensor<Float>(zeros:[distances.shape[0], 2])
        
        // In uniform weighing method each class have same weight.
        // In distance weighing method weight vary based on the inverse of distance.
        if self.weights == "uniform" {
            for i in 0..<distances.shape[0] {
                labelsAndWeightsTensor[i][0] = labels[i]
                labelsAndWeightsTensor[i][1] = Tensor<Float>(1.0)
            }
            return labelsAndWeightsTensor
        } else if self.weights == "distance" {
            var matched = Tensor<Float>(zeros:[1, 2])
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

    /// Returns the individual tensor predicted class.
    ///
    /// - Parameter test: Input tensor to be classified.
    /// - Returns: Predicted class.
    internal func predictSingleSample(_ test: Tensor<Float>) -> Tensor<Float> {

        var distances = Tensor<Float>(zeros: [self.data.shape[0]])
        var maxLabel = Tensor<Float>(zeros: [self.neighbors])
        
        var maxDistances: Tensor<Float>
        var maxIndex: Tensor<Int32>
      
        // Calculate the distance between test and all data points.
        for i in 0..<self.data.shape[0] {
            distances[i] = minkowskiDistance(self.data[i], test, self.p)
        }

        // Find the top neighbors with minimum distance.
        (maxDistances, maxIndex) =
            Raw.topKV2(distances, k: Tensor<Int32>(Int32(data.shape[0])), sorted: true)
        maxDistances = Raw.reverse(maxDistances, dims: Tensor<Bool>([true]))
        maxDistances = maxDistances
            .slice(lowerBounds: Tensor<Int32>([0]), sizes: Tensor<Int32>([Int32(self.neighbors)]))

        maxIndex = Raw.reverse(maxIndex, dims: Tensor<Bool>([true]))
        maxIndex = maxIndex
            .slice(lowerBounds: Tensor<Int32>([0]), sizes: Tensor<Int32>([Int32(self.neighbors)]))

        for i in 0..<self.neighbors {
            maxLabel[i] = self.labels[Int(maxIndex[i].scalarized())]
        }

        /// Weights the neighbors based on their weighing method.
        let labelsAndWeightsTensor = computeWeights(distances: maxDistances, labels: maxLabel)

        var classes: Tensor<Int32>
        var indices: Tensor<Int32>
        (classes, indices) = Raw.unique(Tensor<Int32>(maxLabel))
        
        var kClasses = Tensor<Float>(zeros: [classes.shape[0]])
        var kWeights = Tensor<Float>(zeros: [classes.shape[0]])

        // Add weights based on their neighbors class.
        for i in 0..<labelsAndWeightsTensor.shape[0] {
            for j in 0..<classes.shape[0] {
                if labelsAndWeightsTensor[i][0] == Tensor<Float>(classes[j]) {
                    kClasses[j] = Tensor<Float>(classes[j])
                    kWeights[j] = kWeights[j] + labelsAndWeightsTensor[i][1]
                }
            }
        }
        
        // Return class with heighest weight.
        let resultSet = Raw.topKV2(kWeights, k: Tensor<Int32>(1), sorted: true)
        let classIndex = Int(resultSet.indices[0].scalarized())
        return kClasses[classIndex]
    }

    /// Returns classified test tensor.
    ///
    /// - Parameter data: data Tensor<Float> of shape [number of samples, number of features].
    /// - Returns: classified class tensor.  
    public func prediction(for data: Tensor<Float>) -> Tensor<Float> {

        precondition(data.shape[0] > 0, "data must be non-empty.")

        var predictions = Tensor<Float>(zeros: [data.shape[0]])
        for i in 0..<data.shape[0] {
            predictions[i] = predictSingleSample(data[i])
        }
        return predictions
    }
    
    /// Returns Predict class labels for input samples.
    ///
    /// - Parameters
    ///   - data: Sample tensor of shape [number of samples, number of features].
    ///   - labels: Target label tensor of shape [number of samples].
    /// - Returns: Returns the mean accuracy on the given test data and labels.
    public func score(data: Tensor<Float>, labels: Tensor<Float>) -> Float {
        
        precondition(data.shape[0] == labels.shape[0],
            "data and labels must have same number of samples.")
        precondition(data.shape[0] > 0, "data must be non-empty.")
        precondition(labels.shape[0] > 0, "labels must be non-empty.")

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