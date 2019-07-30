import TensorFlow

/// K neighbors regressor.
///
/// The target is predicted by local interpolation of the targets associated of the nearest
/// neighbors in the training set.
public class KNeighborsRegressor {

    var neighbors: Int
    var p: Int
    var weights: String
    var data: Tensor<Float>
    var labels: Tensor<Float>
  
    /// Create a K neighbors regressor model.
    ///
    /// - Parameters
    ///   - neighbors: Number of neighbors to use, default to `5`.
    ///   - weights: Weight function used in prediction. Possible values `uniform` - uniform
    ///     weighted, `distance` - weight point by inverse of thier distance. Default set to
    ///     `distance`.
    ///   - p: The order of the norm of the difference: `||a - b||_p`, default set to `2`.
    public init(
        neighbors: Int = 5,
        weights: String = "distance",
        p: Int = 2
    ) {
        precondition(neighbors > 1, "Neighbors must be greater than one.")
        precondition(weights == "uniform" || weights == "distance",
            "weights must be either 'uniform' or 'distance'.")
        precondition(p > 1, "p must be greater than one.")

        self.neighbors = neighbors
        self.weights = weights
        self.p = p
        self.data = Tensor<Float>([0.0])
        self.labels = Tensor<Float>([0.0])
    }
  
    /// fit Kneighbors regressor model.
    ///
    /// - Parameters
    ///   - data: Training data tensor of shape [number of samples, number of features].
    ///   - labels: Target value tensor of shape [number of samples].
    public func fit(data: Tensor<Float>, labels: Tensor<Float>) {

        precondition(data.shape[0] == labels.shape[0],
            "data and labels must have same number of samples.")
        precondition(data.shape[0] > 0, "data must be non-empty.")
        precondition(data.shape[1] >= 1, "data must have atleast single feature.")
        precondition(labels.shape[0] > 0, "labels must be non-empty.")

        self.data = data
        self.labels = labels
    }

  
    /// Return the average value of target.
    ///
    /// - Parameters
    ///   - distances: Tensor contains the distance between test tensor and top neighbors.
    ///   - labels: Tensor contains the average value of target.
    internal func computeWeights(
        distances: Tensor<Float>,
        labels: Tensor<Float>
    ) -> Tensor<Float> {
      
        var weightsTensor = Tensor<Float>(zeros:[distances.shape[0]])
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
                weightsTensor[i] = Tensor<Float>(1.0 / distances[i])
                result = result + weightsTensor[i] * labels[i]
            }
              
            result = result / weightsTensor.sum()
            return result
        } 
        return Tensor<Float>(0.0)
    } 

    /// Returns the individual tensor predicted value.
    ///
    /// - Parameter data: Input tensor to be regressed.
    /// - Returns: Predicted target value.
    internal func predictSingleSample(_ test: Tensor<Float>) -> Tensor<Float> {

        var distances = Tensor<Float>(zeros: [self.data.shape[0]])
        var maxLabel = Tensor<Float>(zeros: [self.neighbors])
        var maxDistances: Tensor<Float>
        var maxIndex: Tensor<Int32>
        
        // Calculate the distance between test and all data points.
        for i in 0..<self.data.shape[0] {
            distances[i] = minkowskiDistance(self.data[i], test, p: self.p)
        }
        
        // Find the top neighbors with minimum distance.
        (maxDistances, maxIndex) =
            Raw.topKV2(distances, k: Tensor<Int32>(Int32(data.shape[0])) , sorted: true)
        maxDistances = Raw.reverse(maxDistances, dims: Tensor<Bool>([true]))
        maxDistances = maxDistances
            .slice(lowerBounds: Tensor<Int32>([0]), sizes: Tensor<Int32>([Int32(self.neighbors)]))

        maxIndex = Raw.reverse(maxIndex, dims: Tensor<Bool>([true]))
        maxIndex = maxIndex
            .slice(lowerBounds: Tensor<Int32>([0]), sizes: Tensor<Int32>([Int32(self.neighbors)]))

        for i in 0..<self.neighbors {
            maxLabel[i] = self.labels[Int(maxIndex[i].scalarized())]
        }

        /// Average weight based on neighbors weights.
        let avgWeight = computeWeights(distances: maxDistances, labels: maxLabel)
        return avgWeight
    }
  
    /// Returns predicted value of test tensor.
    ///
    /// - Parameter data: Test tensor of shape [number of samples, number of features].
    /// - Returns: Predicted value tensor. 
    public func prediction(for data: Tensor<Float>) -> Tensor<Float> {

        precondition(data.shape[0] > 0, "data must be non-empty.")

        var predictions = Tensor<Float>(zeros: [data.shape[0]])
        for i in 0..<data.shape[0] {
            predictions[i] = predictSingleSample(data[i])
        }
        return predictions
    }
  
    /// Returns the coefficient of determination R^2 of the prediction.
    ///
    /// - Parameters
    ///   - data: Sample tensor of shape [number of samples, number of features].
    ///   - labels: Target value tensor of shape [number of samples].
    /// - Returns: The coefficient of determination R^2 of the prediction.
    public func score(data: Tensor<Float>, labels: Tensor<Float>) -> Float {

        precondition(data.shape[0] == labels.shape[0],
            "data and labels must have same number of samples.")
        precondition(data.shape[0] > 0, "data must be non-empty.")
        precondition(labels.shape[0] > 0, "labels must be non-empty.")

        let predictedLabels = self.prediction(for: data)
        let u = pow((labels - predictedLabels),2).sum() 
        let v = pow((labels - labels.mean()), 2).sum()
        let score = (1 - (u/v))
        return Float(score.scalarized())
    }
}