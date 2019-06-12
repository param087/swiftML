import TensorFlow

/// The generalized inverse of a matrix using its singular-value decomposition (SVD)
///
/// Return the (Moore-Penrose) pseudo-inverse of a matrix
///
/// - Parameter x: The matrix to pseudo-inverse
/// - Returns : Pseudo-inverse matrix
public func pinv(_ x: Tensor<Float>) -> Tensor<Float> {
    let svd = Raw.svd(x)
    let d = Raw.diag(diagonal: svd.s)
    return matmul(matmul(svd.v, Raw.matrixInverse(d)), svd.u.transposed())
}

