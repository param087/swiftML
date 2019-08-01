import TensorFlow

extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// The (Moore-Penrose) pseudoinverse of a matrix using its singular-value decomposition (SVD).
    public var pseudoinverse: Tensor {
        let svd = Raw.svd(self)
        let diag = Raw.diag(diagonal: svd.s)
        return matmul(matmul(svd.v, Raw.matrixInverse(diag)), svd.u.transposed())
    }
}

/// Returns sign-corrected versions of the left and right singular vectors from
/// single value decomposition (SVD).
///
/// - Parameters:
///   - u: The tensor containing the left singular vectors for each matrix.
///   - v: The tensor containing the right singular vectors for each matrix.
/// - Returns: The sign corrected versions of `u` and `v` to ensure
///   deterministic output.
public func svdFlip<Scalar: TensorFlowFloatingPoint>(
    u: Tensor<Scalar>,
    v: Tensor<Scalar>,
    uBasedDecision: Bool = true
) -> (Tensor<Scalar>, Tensor<Scalar>) {
    let signs: Tensor<Scalar>
    if uBasedDecision {
        let maxAbsCols = abs(u).argmax(squeezingAxis: 0)
        var colValueForSign = Tensor<Scalar>(zeros: [u.shape[1]])
        for i in 0..<u.shape[1] {
            colValueForSign[i] = u[Int(maxAbsCols[i].scalarized()), i]
        }
        signs = Raw.sign(colValueForSign)
    } else {
        let maxAbsRows = abs(v).argmax(squeezingAxis: 1)
        var rowValueForSign = Tensor<Scalar>(zeros: [u.shape[0]])
        for i in 0..<u.shape[0] {
            rowValueForSign[i] = v[i, Int(maxAbsRows[i].scalarized())]
        }
        signs = Raw.sign(rowValueForSign)
    }
    return (u * signs, v * signs.reshaped(to: [u.shape[1], 1]))
}

/// Returns the Minkowski distance between two tensors for the given distance metric `p`.
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
/// - Parameters:
///   - a: The first tensor.
///   - b: The second tensor.
/// - Returns: The Euclidean distance: `||a - b||_2`.
public func euclideanDistance<Scalar: TensorFlowFloatingPoint>(
    _ a: Tensor<Scalar>, _ b: Tensor<Scalar>, p: Int
) -> Tensor<Scalar> {
    precondition(a.shape == b.shape, "Both inputs must have the same shape.")
    return minkowskiDistance(a, b, p: 2)
}