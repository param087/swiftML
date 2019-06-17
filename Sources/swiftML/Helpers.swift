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

/// Return Sign correction to ensure deterministic output from SVD.
///
/// - Parameters
///   - u: The tensor containing of left singular vectors for each matrix.
///   - v: The tensor containing of right singular vectors for each matrix.
/// - Returns: Sign correction to ensure deterministic output.
func svdFlip(
    u: Tensor<Float>,
    v: Tensor<Float>,
    uBasedDecision: Bool = true
) -> (Tensor<Float>, Tensor<Float>) {
  
    var U = u
    var V = v
    
    if uBasedDecision {
    
        let maxAbsCols = abs(u).argmax(squeezingAxis: 0)
        var colValueForSign = Tensor<Float>(zeros: [u.shape[1]])
        
        for i in 0..<u.shape[1] {
            colValueForSign[i] = u[Int(maxAbsCols[i].scalarized()), i]
        }
        
        let signs = Raw.sign(colValueForSign)
        U = U * signs
        V = V * signs.reshaped(to: [u.shape[1], 1])
        
    } else {
        
        let maxAbsRows = abs(v).argmax(squeezingAxis: 1)
        var rowValueForSign = Tensor<Float>(zeros:[u.shape[0]])
        
        for i in 0..<u.shape[0] {
            rowValueForSign[i] = v[i, Int(maxAbsRows[i].scalarized())]
        }
        
        let signs = Raw.sign(rowValueForSign)
        U = U * signs
        V = V * signs.reshaped(to: [u.shape[1], 1])
    }
    
    return (U, V)
}