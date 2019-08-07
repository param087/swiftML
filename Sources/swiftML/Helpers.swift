import TensorFlow

extension Tensor where Scalar: TensorFlowFloatingPoint {
    /// The (Moore-Penrose) pseudoinverse of a matrix using its singular-value decomposition (SVD).
<<<<<<< HEAD
=======
    ///
    /// Reference: ["Moore–Penrose inverse"](
    /// https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)
>>>>>>> 068e99a74c4c913f173f9445a95440eab4ee503f
    public var pseudoinverse: Tensor {
        let svd = Raw.svd(self)
        let diag = Raw.diag(diagonal: svd.s)
        return matmul(matmul(svd.v, Raw.matrixInverse(diag)), svd.u.transposed())
    }
}

<<<<<<< HEAD
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
=======
/// Returns deterministic SVD contains sign-corrected versions of left singular vectors and right
/// singular vectors from input matrix.
///
/// Reference: ["Determinitic SVD"](https://github.com/scikit-learn/
/// scikit-learn/blob/53f76d1a24ef42eb8c620fc1116c53db11dd07d9/sklearn/utils/extmath.py#L482)
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
    var (s, u, v) = Raw.svd(input)
    v = v.transposed()

    let signs: Tensor<T>
    
    if columnBasedSignFlipping {
        let maxAbsCols = abs(u).argmax(squeezingAxis: 0)
        var colValueForSign = Tensor<T>(zeros: [u.shape[1]])
>>>>>>> 068e99a74c4c913f173f9445a95440eab4ee503f
        for i in 0..<u.shape[1] {
            colValueForSign[i] = u[Int(maxAbsCols[i].scalarized()), i]
        }
        signs = Raw.sign(colValueForSign)
    } else {
        let maxAbsRows = abs(v).argmax(squeezingAxis: 1)
<<<<<<< HEAD
        var rowValueForSign = Tensor<Scalar>(zeros: [u.shape[0]])
=======
        var rowValueForSign = Tensor<T>(zeros: [u.shape[0]])
>>>>>>> 068e99a74c4c913f173f9445a95440eab4ee503f
        for i in 0..<u.shape[0] {
            rowValueForSign[i] = v[i, Int(maxAbsRows[i].scalarized())]
        }
        signs = Raw.sign(rowValueForSign)
    }
<<<<<<< HEAD
    return (u * signs, v * signs.reshaped(to: [u.shape[1], 1]))
=======
    return (s, u * signs, v * signs.reshaped(to: [u.shape[1], 1]))
>>>>>>> 068e99a74c4c913f173f9445a95440eab4ee503f
}

/// Returns the Minkowski distance between two tensors for the given distance metric `p`.
///
<<<<<<< HEAD
=======
/// Reference: ["Minkowski distance"](https://en.wikipedia.org/wiki/Minkowski_distance)
///
>>>>>>> 068e99a74c4c913f173f9445a95440eab4ee503f
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
<<<<<<< HEAD
=======
/// Reference: ["Euclidean distance"](https://en.wikipedia.org/wiki/Euclidean_distance)
///
>>>>>>> 068e99a74c4c913f173f9445a95440eab4ee503f
/// - Parameters:
///   - a: The first tensor.
///   - b: The second tensor.
/// - Returns: The Euclidean distance: `||a - b||_2`.
public func euclideanDistance<Scalar: TensorFlowFloatingPoint>(
    _ a: Tensor<Scalar>, _ b: Tensor<Scalar>, p: Int
) -> Tensor<Scalar> {
    precondition(a.shape == b.shape, "Both inputs must have the same shape.")
    return minkowskiDistance(a, b, p: 2)
<<<<<<< HEAD
}
=======
}
>>>>>>> 068e99a74c4c913f173f9445a95440eab4ee503f
