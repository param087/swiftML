import TensorFlow

/// Returns deterministic SVD contains sign-corrected versions of left singular vectors and right
/// singular vectors from input matrix.
///
/// Reference: ["Determinitic SVD"](
/// https://github.com/scikit-learn/scikit-learn/blob/53f76d1a24ef42eb8c620fc1116c53db11dd07d9/sklearn/utils/extmath.py#L482)
///
/// - Parameters:
///   - input: The input matrix.
///   - columnBasedSignFlipping - If `true`, use the columns of `u` as the basis for sign flipping.
///     Otherwise, use the rows of v. The choice of which variable to base the decision on is
///     generally algorithm dependent, default to `true`.
/// - Returns: The sign corrected svd to ensure deterministic output.
public func deterministicSvd<T: FloatingPoint & TensorFlowScalar>(
    _ input: Tensor<T>,
    columnBasedSignFlipping: Bool = true
) -> (s: Tensor<T>, u: Tensor<T>, v: Tensor<T>) {
    var (s, u, v) = _Raw.svd(input)
    v = v.transposed()

    let signs: Tensor<T>
    
    if columnBasedSignFlipping {
        let maxAbsCols = abs(u).argmax(squeezingAxis: 0)
        var colValueForSign = Tensor<T>(zeros: [u.shape[1]])
        for i in 0..<u.shape[1] {
            colValueForSign[i] = u[Int(maxAbsCols[i].scalarized()), i]
        }
        signs = _Raw.sign(colValueForSign)
    } else {
        let maxAbsRows = abs(v).argmax(squeezingAxis: 1)
        var rowValueForSign = Tensor<T>(zeros: [u.shape[0]])
        for i in 0..<u.shape[0] {
            rowValueForSign[i] = v[i, Int(maxAbsRows[i].scalarized())]
        }
        signs = _Raw.sign(rowValueForSign)
    }
    return (s, u * signs, v * signs.reshaped(to: [u.shape[1], 1]))
}

/// Returns the Minkowski distance between two tensors for the given distance metric `p`.
///
/// Reference: ["Minkowski distance"](https://en.wikipedia.org/wiki/Minkowski_distance)
///
/// - Parameters:
///   - a: The first tensor.
///   - b: The second tensor.
///   - p: The order of the norm of the difference: `||a - b||_p`.
/// - Returns: The Minkowski distance based on value of `p`.
public func minkowskiDistance<Scalar: TensorFlowFloatingPoint>(
    _ a: Tensor<Scalar>, _ b: Tensor<Scalar>, p: Int
) -> Tensor<Scalar> {
    precondition(a.shape == b.shape, "Both inputs must have the same shape.")
    precondition(p > 0, "p must be positive.")
    return pow(pow(abs(b - a), Scalar(p)).sum(), 1 / Scalar(p))
}

/// Returns the Euclidean distance between two tensors.
///
/// Reference: ["Euclidean distance"](https://en.wikipedia.org/wiki/Euclidean_distance)
///
/// - Parameters:
///   - a: The first tensor.
///   - b: The second tensor.
/// - Returns: The Euclidean distance: `||a - b||_2`.
public func euclideanDistance<Scalar: TensorFlowFloatingPoint>(
    _ a: Tensor<Scalar>, _ b: Tensor<Scalar>
) -> Tensor<Scalar> {
    precondition(a.shape == b.shape, "Both inputs must have the same shape.")
    return minkowskiDistance(a, b, p: 2)
}