import TensorFlow

/// K neighbors classifier.
///
/// Classifier implementing the k-nearest neighbors vote.
public class KNeighborsClassifier {

    var neighbors: Int
    var p: Int
    var weights: String
    var X: Tensor<Float>
    var y: Tensor<Float>
  
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
        self.X = Tensor<Float>([0.0])
        self.y = Tensor<Float>([0.0])
    }
  
    /// fit Kneighbors classifier model.
    ///
    /// - Parameters
    ///   - X: Training data Tensor<Float> of shape [number of samples, number of features].
    ///   - y: Target value Tensor<Float> of shape [number of samples].
    public func fit(X: Tensor<Float>, y: Tensor<Float>) {

        precondition(X.shape[0] == y.shape[0], "X and y must have same number of samples.")
        precondition(X.shape[0] > 0, "X must be non-empty.")
        precondition(X.shape[1] >= 1, "X must have atleast single feature.")
        precondition(y.shape[0] > 0, "y must be non-empty.")

        self.X = X
        self.y = y
    }
  
    /// Return the minkowski distance based on value of p.
    ///
    /// - Parameters
    ///   - a: Input tensor to find distance.
    ///   - b: Input tensor to find distance.
    /// - Returns: Minkowski distance based on value of p.
    public func distance(a: Tensor<Float>, b: Tensor<Float>) -> Tensor<Float> {
        return pow(pow(abs(b - a), Float(self.p)).sum(), 1.0 / Float(self.p))
    }

    /// Return the weights of each neighbors.
    ///
    /// - Parameters
    ///   - distances: Tensor contains the distance between test tensor and top neighbors.
    ///   - labels: Tensor contains the classes of neighbors.
    public func computeWeights(distances: Tensor<Float>, labels: Tensor<Float>) -> Tensor<Float> {
      
        var weightsV = Tensor<Float>(zeros:[distances.shape[0], 2])
        
        // In uniform weighing method each class have same weight.
        // In distance weighing method weight vary based on the inverse of distance.
        if self.weights == "uniform" {
            for i in 0..<distances.shape[0] {
                weightsV[i][0] = labels[i]
                weightsV[i][1] = Tensor<Float>(1.0)
            }
            return weightsV
        } else if self.weights == "distance" {
            var matched = Tensor<Float>(zeros:[1, 2])
            for i in 0..<distances.shape[0] {
                if distances[i] == Tensor<Float>([0.0]) {
                    matched[0][0] = labels[i]
                    matched[0][1] = Tensor<Float>(1.0)
                    return matched
                }
                weightsV[i][0] = labels[i]
                weightsV[i][1] = Tensor<Float>(1.0/distances[i])
            }
            return weightsV
        }  
        return Tensor<Float>(0.0)
    } 

    /// Returns the individual tensor predicted class.
    ///
    /// - Parameter test: Input tensor to be classified.
    /// - Returns: Predicted class.
    public func predictOne(test: Tensor<Float>) -> Tensor<Float> {

        var distances = Tensor<Float>(zeros: [self.X.shape[0]])
        var maxLabel = Tensor<Float>(zeros: [self.neighbors])
        var maxDistances: Tensor<Float>
        var maxIndex: Tensor<Int32>
      
        // Calculate the distance between test and all data points.
        for i in 0..<self.X.shape[0] {
            distances[i] = self.distance(a: self.X[i], b: test)
        }

        // Find the top neighbors with minimum distance.
        (maxDistances, maxIndex) =
            Raw.topKV2(distances, k: Tensor<Int32>(Int32(X.shape[0])), sorted: true)
        maxDistances = Raw.reverse(maxDistances, dims: Tensor<Bool>([true]))
        maxDistances = maxDistances
            .slice(lowerBounds: Tensor<Int32>([0]), sizes: Tensor<Int32>([Int32(self.neighbors)]))

        maxIndex = Raw.reverse(maxIndex, dims: Tensor<Bool>([true]))
        maxIndex = maxIndex
            .slice(lowerBounds: Tensor<Int32>([0]), sizes: Tensor<Int32>([Int32(self.neighbors)]))

        for i in 0..<self.neighbors {
            maxLabel[i] = self.y[Int(maxIndex[i].scalarized())]
        }

        /// Weights the neighbors based on their weighing method.
        let weightsV = computeWeights(distances: maxDistances, labels: maxLabel)

        var cls: Tensor<Int32>
        var idx: Tensor<Int32>
        (cls, idx) = Raw.unique(Tensor<Int32>(maxLabel))
        
        var kClasses = Tensor<Float>(zeros: [cls.shape[0]])
        var kWeights = Tensor<Float>(zeros: [cls.shape[0]])

        // Add weights based on their neighbors class.
        for i in 0..<weightsV.shape[0] {
            for j in 0..<cls.shape[0] {
                if weightsV[i][0] == Tensor<Float>(cls[j]) {
                    kClasses[j] = Tensor<Float>(cls[j])
                    kWeights[j] = kWeights[j] + weightsV[i][1]
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
    /// - Parameter X: Test Tensor<Float> of shape [number of samples, number of features].
    /// - Returns: classified class tensor.  
    public func predict(X: Tensor<Float>) -> Tensor<Float> {

        precondition(X.shape[0] > 0, "X must be non-empty.")

        var predictions = Tensor<Float>(zeros: [X.shape[0]])
        for i in 0..<X.shape[0] {
            predictions[i] = predictOne(test: X[i])
        }
        return predictions
    }
    
    /// Returns Predict class labels for input samples.
    ///
    /// - Parameters
    ///   - X: Sample tensor of shape [number of samples, number of features].
    ///   - y: Target label tensor of shape [number of samples].
    /// - Returns: Returns the mean accuracy on the given test data and labels.
    public func score(X: Tensor<Float>, y: Tensor<Float>) -> Float {
        
        precondition(X.shape[0] == y.shape[0], "X and y must have same number of samples.")
        precondition(X.shape[0] > 0, "X must be non-empty.")
        precondition(y.shape[0] > 0, "y must be non-empty.")

        let yPred = self.predict(X: X)
        var count:Int = 0
        for i in 0..<X.shape[0] {
            if yPred[i] == y[i] {
                count = count + 1
            }
        }
        let score = Float(count) / Float(y.shape[0])
        return score
    }
}