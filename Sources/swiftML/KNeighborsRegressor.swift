import TensorFlow

/// K neighbors regressor.
///
/// The target is predicted by local interpolation of the targets associated of the nearest
/// neighbors in the training set.
public class KNeighborsRegressor {
    /// The order of the norm of the difference: `||a - b||_p`.
    var p: Int
    /// Weight function used in prediction.
    public var weights: String
    /// Number of neighbors.
    var neighborCount: Int
    /// The training data.
    var data: Tensor<Float>
    /// The target value correspoing to training data.
    var labels: Tensor<Float>
  
    /// Create a K neighbors regressor model.
    ///
    /// - Parameters:
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
        precondition(p > 0, "p must be positive.")

        self.neighborCount = neighborCount
        self.weights = weights
        self.p = p
        self.data = Tensor<Float>([0.0])
        self.labels = Tensor<Float>([0.0])
    }
  
    /// Fit a K-neighbors regressor model.
    ///
    /// - Parameters:
    ///   - data: Training data with shape `[sample count, feature count]`.
    ///   - labels: Target value with shape `[sample count]`.
    public func fit(data: Tensor<Float>, labels: Tensor<Float>) {
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")
        precondition(data.shape[1] >= 1,
            "Data must have feature count greater than or equal to one.")
        precondition(labels.shape[0] > 0, "Labels must have a positive sample count.")
        precondition(data.shape[0] == labels.shape[0],
            "Data and labels must have the same sample count.")

        self.data = data
        self.labels = labels
    }
  
    /// Returns the average value of target.
    ///
    /// - Parameters:
    ///   - distances: Contains the distance between test data and top neighbors.
    ///   - labels: Contains the value of target.
    internal func computeWeights(
        distances: Tensor<Float>,
        labels: Tensor<Float>
    ) -> Tensor<Float> {
        var weightsTensor = Tensor<Float>(zeros: [distances.shape[0]])
        var result = Tensor<Float>(0.0)
        
        // In `uniform` weighing method each class have same weight.
        // In `distance` weighing method weight vary based on the inverse of distance.
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

    /// Returns the individual predicted value.
    ///
    /// - Parameters:
    ///   - data: Input data to be regressed.
    /// - Returns: Predicted target value.
    internal func predictSingleSample(_ test: Tensor<Float>) -> Tensor<Float> {
        var distances = Tensor<Float>(zeros: [self.data.shape[0]])
        var minDistanceLabels = Tensor<Float>(zeros: [self.neighborCount])
        var minDistances: Tensor<Float>
        var minDistanceIndex: Tensor<Int32>
        
        // Calculate the distance between test and all data points.
        for i in 0..<self.data.shape[0] {
            distances[i] = minkowskiDistance(self.data[i], test, p: self.p)
        }
        
        // Find the top neighbors with minimum distance.
        (minDistances, minDistanceIndex) =
            Raw.topKV2(distances, k: Tensor<Int32>(Int32(data.shape[0])), sorted: true)
        minDistances = Raw.reverse(minDistances, dims: Tensor<Bool>([true]))
        minDistances = minDistances
            .slice(lowerBounds: Tensor<Int32>([0]),
                sizes: Tensor<Int32>([Int32(self.neighborCount)]))

        minDistanceIndex = Raw.reverse(minDistanceIndex, dims: Tensor<Bool>([true]))
        minDistanceIndex = minDistanceIndex
            .slice(lowerBounds: Tensor<Int32>([0]),
                sizes: Tensor<Int32>([Int32(self.neighborCount)]))

        for i in 0..<self.neighborCount {
            minDistanceLabels[i] = self.labels[Int(minDistanceIndex[i].scalarized())]
        }

        // Average weight based on neighbors weights.
        let avgWeight = computeWeights(distances: minDistances, labels: minDistanceLabels)
        return avgWeight
    }
  
    /// Returns predicted values.
    ///
    /// - Parameters:
    ///   - data: Test data with shape `[sample count, feature count]`.
    /// - Returns: Predicted value tensor.
    public func prediction(for data: Tensor<Float>) -> Tensor<Float> {
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")

        var predictions = Tensor<Float>(zeros: [data.shape[0]])
        for i in 0..<data.shape[0] {
            predictions[i] = predictSingleSample(data[i])
        }
        return predictions
    }
  
    /// Returns the coefficient of determination (`R^2`) of the prediction.
    ///
    /// - Parameters:
    ///   - data: Sample data with shape `[sample count, feature count]`.
    ///   - labels: Target values with shape `[sample count]`.
    /// - Returns: The coefficient of determination (`R^2`) of the prediction.
    public func score(data: Tensor<Float>, labels: Tensor<Float>) -> Float {
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")
        precondition(labels.shape[0] > 0, "Labels must have a positive sample count.")
        precondition(data.shape[0] == labels.shape[0],
            "Data and labels must have the same sample count.")

        let predictedLabels = self.prediction(for: data)
        let u = pow((labels - predictedLabels),2).sum() 
        let v = pow((labels - labels.mean()), 2).sum()
        let score = (1 - (u/v))
        return Float(score.scalarized())
    }
}