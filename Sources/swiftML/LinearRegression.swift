import TensorFlow

/// Linear Regression.
///
/// Reference: ["Linear regression"](https://en.wikipedia.org/wiki/Linear_regression)
public protocol LinearRegression {
    /// Whether to calculate the intercept for this model.
    var fitIntercept: Bool {get set}
    /// The weights of the model.
    var weights: Tensor<Float> {get set}

    /// Fit a linear model.
    mutating func fit(data: Tensor<Float>, labels: Tensor<Float>)
    /// Returns prediction using linear model.
    mutating func prediction(for data: Tensor<Float>) -> Tensor<Float>
    /// Returns the coefficient of determination (`R^2`) of the prediction.
    mutating func score(data: Tensor<Float>, labels: Tensor<Float>) -> Float
}

extension LinearRegression {

    /// Returns prediction using linear model.
    ///
    /// - Parameter data: Input data with shape `[sample count, feature count]`.
    /// - Returns: Predicted output.
    public func prediction(for data: Tensor<Float>) -> Tensor<Float> {
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")

        var data: Tensor<Float> = data
        if self.fitIntercept {
            let ones = Tensor<Float>(ones: [data.shape[0], 1])
            data = ones.concatenated(with: data, alongAxis: -1)
        }
        let predictedLabels = matmul(data, self.weights)
        return predictedLabels
    }

    /// Returns the coefficient of determination (`R^2`) of the prediction.
    ///
    /// - Parameters
    ///   - data: Sample data with shape `[sample count, feature count]`.
    ///   - labels: Target value with shape `[sample count, 1]`.
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