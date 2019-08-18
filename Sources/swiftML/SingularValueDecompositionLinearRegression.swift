import TensorFlow

/// Singular value decomposition linear regression.
///
/// Reference: ["Use of the Singular Value Decomposition in Regression Analysis"](
/// https://pdfs.semanticscholar.org/aef2/68c21be034bfd6228bf3946cb46e3c62cdb1.pdf)
public class SingularValueDecompositionLinearRegression: LinearRegression {
    /// Whether to calculate the intercept for this model.
    public var fitIntercept: Bool
    /// The weights of the model.
    public var weights: Tensor<Float>
    
    /// Creates a linear regression model.
    ///
    /// - Parameters:
    ///   - fitIntercept: Whether to calculate the intercept for this model. If `false`, no
    ///     intercept will be used in calculations. The default is `true`.
    public init(
        fitIntercept: Bool = true
    ) {
        self.fitIntercept = fitIntercept
        self.weights = Tensor<Float>(0)
    }

    /// Fit a linear model.
    ///
    /// - Parameters:
    ///   - data: Training data with shape `[sample count, feature count]`.
    ///   - labels: Target value with shape `[sample count, target count]`.
    public func fit(data: Tensor<Float>, labels: Tensor<Float>) {
        precondition(data.shape[0] > 0, "Data must have a positive sample count.")
        precondition(data.shape[1] >= 1,
            "Data must have feature count greater than or equal to one.")
        precondition(labels.shape[0] > 0, "Labels must have a positive sample count.")
        precondition(labels.shape[1] >= 1,
            "Labels must have target feature count greater than or equal to one.")
        precondition(data.shape[0] == labels.shape[0],
            "Data and labels must have the same sample count.")

        var data: Tensor<Float> = data
        if self.fitIntercept {
            let ones = Tensor<Float>(ones: [data.shape[0], 1])
            data = ones.concatenated(with: data, alongAxis: -1)
        }   

        let svd = Raw.svd(matmul(data.transposed(), data))
        let s = Raw.diag(diagonal: svd.s)
        let dataSqRegInv = matmul(matmul(svd.v, s.pseudoinverse), svd.u.transposed())
        self.weights = matmul(matmul(dataSqRegInv, data.transposed()), labels)
    }
}
