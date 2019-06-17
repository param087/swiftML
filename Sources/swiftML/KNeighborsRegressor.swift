import TensorFlow

/// K neighbors regressor.
///
/// The target is predicted by local interpolation of the targets associated of the nearest
/// neighbors in the training set.
public class KNeighborsRegressor {

    public var neighbors: Int
    public var p: Int
    public var weights: String
    public var X: Tensor<Float>
    public var y: Tensor<Float>
  
    /// Create a K neighbors regressor model.
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
  
    /// fit Kneighbors regressor model.
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

        return pow(pow(abs(b - a), Float(self.p)).sum(), 1.0/Float(self.p))
    }
  
    /// Return the average value of target.
    ///
    /// - Parameters
    ///   - distances: Tensor contains the distance between test tensor and top neighbors.
    ///   - labels: Tensor cthe average value of target.
    public func computeWeights(distances: Tensor<Float>, labels: Tensor<Float>) -> Tensor<Float> {
      
        var weightsV = Tensor<Float>(zeros:[distances.shape[0]])
        var result = Tensor<Float>(0.0)
        
        // In uniform weighing method each class have same weight.
        // In distance weighing method weight vary based on the inverse of distance.
        if self.weights == "uniform" {
            return labels.mean()
        } else if self.weights == "distance" {
            
            for i in 0..<distances.shape[0] {
                if distances[i] == Tensor<Float>([0.0]) {
                    return labels[i]
                }
                weightsV[i] = Tensor<Float>(1.0/distances[i])
                result = result + weightsV[i] * labels[i]
            }
              
            result = result / weightsV.sum()
            return result
        } 
        return Tensor<Float>(0.0)
    } 

    /// Returns the individual tensor predicted value.
    ///
    /// - Parameter test: Input tensor to be regressed.
    /// - Returns: Predicted target value.
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
            Raw.topKV2(distances, k: Tensor<Int32>(Int32(X.shape[0])) , sorted: true)
        maxDistances = Raw.reverse(maxDistances, dims: Tensor<Bool>([true]))
        maxDistances = maxDistances
            .slice(lowerBounds: Tensor<Int32>([0]), sizes: Tensor<Int32>([Int32(self.neighbors)]))

        maxIndex = Raw.reverse(maxIndex, dims: Tensor<Bool>([true]))
        maxIndex = maxIndex
            .slice(lowerBounds: Tensor<Int32>([0]), sizes: Tensor<Int32>([Int32(self.neighbors)]))

        for i in 0..<self.neighbors {
            maxLabel[i] = self.y[Int(maxIndex[i].scalarized())]
        }

        /// Average weight based on neighbors weights.
        let avgWeight = computeWeights(distances: maxDistances, labels: maxLabel)
        return avgWeight
    }
  
    /// Returns predicted value of test tensor.
    ///
    /// - Parameter X: Test Tensor<Float> of shape [number of samples, number of features].
    /// - Returns: Predicted value tensor. 
    public func predict(X: Tensor<Float>) -> Tensor<Float> {

        precondition(X.shape[0] > 0, "X must be non-empty.")

        var predictions = Tensor<Float>(zeros: [X.shape[0]])
        for i in 0..<X.shape[0] {
            predictions[i] = predictOne(test: X[i])
        }
        return predictions
    }
  
    /// Returns the coefficient of determination R^2 of the prediction.
    ///
    /// - Parameters
    ///   - X: Sample tensor of shape [number of samples, number of features].
    ///   - y: Target value tensor of shape [number of samples].
    /// - Returns: The coefficient of determination R^2 of the prediction.
    public func score(X: Tensor<Float>, y: Tensor<Float>) -> Float {

        precondition(X.shape[0] == y.shape[0], "X and y must have same number of samples.")
        precondition(X.shape[0] > 0, "X must be non-empty.")
        precondition(y.shape[0] > 0, "y must be non-empty.")

        let y_pred = self.predict(X: X)
        let u = pow((y - y_pred),2).sum() 
        let v = pow((y - y.mean()), 2).sum()
        let score = (1 - (u/v))
        return Float(score.scalarized())
    }
}