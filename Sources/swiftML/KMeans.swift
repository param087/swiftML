import TensorFlow

/// K-Means Clustering
public class KMeans {
    
    public var nClusters: Int
    public var maxIterations: Int
    // Centroids of Clsuters.
    public var centroids : Tensor<Float>
    // Inertia is the sum of square distances of samples to their closest cluster center.
    public var inertia: Tensor<Float>
    // Predicted cluster labels.
    public var labels: Tensor<Int32>
    public var initializer: String
    public var seed: Int64
  
    /// Creates a logistic regression model.
    ///
    /// - Parameters
    ///   - nClusters: The number of clusters to form as well as the number of centroids to
    ///     generate, default to 2.
    ///   - maxIterations: Maximum number of iterations of the k-means algorithm to run, default
    ///     to 300.
    ///   - initializer: Select the initialization method for centroids. 'kmean++', 'random'
    ///     methods for initialization, default to 'kmean++'.
    ///   - seed: Used to initialize a pseudorandom number generator.
    public init(
        nClusters: Int = 2,
        maxIterations: Int = 300,
        initializer: String = "kmean++",
        seed: Int64 = 0
    ) {

        precondition(nClusters > 1, "Number of clusters must be greater than one.")
        precondition(maxIterations >= 0, "Maximum number of Iterations must be non-ngative.")
        precondition(initializer == "kmean++" || initializer == "random",
            "intialializer must be 'keman++' or 'random'.")
        
        self.nClusters = nClusters
        self.maxIterations = maxIterations
        self.initializer = initializer
        self.seed = seed
        self.centroids = Tensor<Float>([0.0])
        self.inertia = Tensor<Float>(0)
        self.labels = Tensor<Int32>([0])
        
    }
    
    /// Returns the euclidean distance.
    ///
    /// - Parameters
    ///   - a: Tensor<Float> of shape [1, number of features].
    ///   - b: Tensor<Float> of shape [1, number of features].
    /// - Returns: The euclidean distace.
    public func euclideanDistance(a: Tensor<Float>, b: Tensor<Float>) -> Tensor<Float> {
        let dist = pow((a - b), 2).sum()
        return dist
    }
  
    /// Return the index of minimum euclidean distance between Tensors.
    ///
    /// - Parameters
    ///   - centrodis: Tensor<Float> of shape [1, number of features].
    ///   - x: Tensor<Float> of shape [number of sample, number of features].
    /// - Returns: Index of minimum euclidean distance tensor in x.
    public func nearest(centroids: Tensor<Float>, x: Tensor<Float>) -> Tensor<Int32> {
        var distances = Tensor<Float>(zeros: [centroids.shape[0], 1])
        for i in 0..<centroids.shape[0] {
            distances[i][0] = self.euclideanDistance(a: centroids[i], b: x)
        }
        return distances.argmin()
    }

    /// Heuristic Initialization of centroids.
    ///
    /// - Parameter X: Tensor<Float> of shape [number of sample, number of features].
    public func kmeanPP(X: Tensor<Float>) {
        
        var distance = Tensor<Float>(zeros:[X.shape[0], 1])
        
        // Choose first center uniformly at random from among the data points.
        let index:Int = Int.random(in: 0..<X.shape[0])
        self.centroids[0] = X[index]

        // For each data point x, compute D(x), the distance between x and the nearest center that
        // has already been chosen.
        // Choose one new data point at random as a new center, using a weighted probability
        // distribution where a point x is chosen with probability proportional to D(x)2.
        for i in 1..<self.nClusters {

            for x_ind in 0..<X.shape[0] {
                var d = Tensor<Float>(zeros:[i, 1])
                for c_ind in 0..<i {
                    d[c_ind][0] = euclideanDistance(a:self.centroids[c_ind], b:X[x_ind])
                }
                distance[x_ind][0] = d.min()
            }

            let prob = distance/distance.sum()
            let cumsum = Raw.cumsum(prob.flattened(), axis: Tensor<Int32>(-1))
            let rand = Tensor<Float>(Float.random(in: 0...1))

            var idx:Int = 0

            for e_id in 0..<cumsum.shape[0] {
                if cumsum[e_id] >= rand {
                    break
                } else {
                    idx = idx + 1
                }
            }
            self.centroids[i] = X[idx]
        }
    }
    
    /// Random Initialization of centroids.
    ///
    /// - Parameter X: Tensor<Float> of shape [number of sample, number of features].
    public func randomInitializer(X: Tensor<Float>) {
        
        /// shuffle the input Tensor
        let shuffled = Raw.randomShuffle(value: X, seed: self.seed)
        for i in 0..<nClusters {
            self.centroids[i] = shuffled[i]
        }
    }
    
    /// Compute k-means clustering.
    ///
    /// - Parameter X: Input Tensor<Float> of shape [number of sample, number of features].
    public func fit(X: Tensor<Float>) {
        
        precondition(X.shape[0] > 0, "X must be non-empty.")

        // reshape centroid of required shape based on input Tensor and number of clusters.
        self.centroids = Tensor<Float>(zeros:[nClusters, X.shape[1]])
        self.labels = Tensor<Int32>(zeros:[X.shape[0],1])
        
        if self.initializer == "kmean++" {
            self.kmeanPP(X: X)
        } else if self.initializer == "random" {
            self.randomInitializer(X: X)
        }
        
        for _ in 0..<self.maxIterations {
            
            var indices = [[Tensor<Int32>]]()
            
            for i in 0..<X.shape[0] {
                self.labels[i][0] = self.nearest(centroids: self.centroids, x: X[i])
            }
            
            for i in 0..<self.nClusters {
                var elementSet = [Tensor<Int32>]()
                for j in 0..<self.labels.shape[0] {
                    if self.labels[j].scalarized() == i {
                        let element = Tensor<Int32>(Int32(j))
                        elementSet.append(element)
                    }
                }
                indices.append(elementSet)
            }

            // Update the cluster centroids based on average of element in cluster. 
            for i in 0..<indices.count {
                var temp = Tensor<Float>(zeros:[1, X.shape[1]])
                for j in 0..<indices[i].count {
                    let id:Int = Int(indices[i][j].scalarized())
                    temp = temp + X[id]
                }
                
                temp = temp * Tensor<Float>((Float(1) / Float(indices[i].count)))
                self.centroids[i] = temp[0]
            }
        }
        
        // sum of square distances from the closest cluster
        for i in 0..<X.shape[0] {
            self.inertia = self.inertia + 
                pow((self.centroids[Int(self.labels[i].scalarized())] - X[i]), 2).sum()
        }       
    }
  
    /// Returns the prediced cluster labels.
    ///
    /// - Returns: Predicted label Tensor<Int32>.
    public func predict() -> Tensor<Int32> {
        return self.labels
    }
  
    /// Returns fit and prediced cluster labels.
    ///
    /// - Parameter X: Input Tensor<Float> of shape [number of sample, number of features].
    /// - Returns: Predicted label Tensor<Int32>.
    public func fitPredict(X: Tensor<Float>) -> Tensor<Int32> {

        precondition(X.shape[0] > 0, "X must be non-empty.")
        
        self.fit(X: X)
        return self.predict()
    }
    
    /// Returns Transform input to a cluster-distance space.
    ///
    /// - Parameter X: Input Tensor<Float> of shape [number of sample, number of features].
    /// - Returns: Transformed X to a cluster-distance space.
    public func transform(X: Tensor<Float>) -> Tensor<Float> {
        
        precondition(X.shape[0] > 0, "X must be non-empty.")

        var transMat = Tensor<Float>(zeros:[X.shape[0], self.nClusters])
        
        for i in 0..<X.shape[0] {
            for j in 0..<self.nClusters {
                transMat[i][j] = self.euclideanDistance(a: X[i], b:self.centroids[j])
            }
        }
        return transMat
    }

    /// Returns fit and Transform input to a cluster-distance space.
    ///
    /// - Parameter X: Input Tensor<Float> of shape [number of sample, number of features].
    /// - Returns: Transformed X to a cluster-distance space.
    public func fitTransform(X: Tensor<Float>) -> Tensor<Float> {

        precondition(X.shape[0] > 0, "X must be non-empty.")

        self.fit(X: X)
        return self.transform(X: X)
    }
    
    /// Returns the sum of square distances of samples to their closest cluster center.
    ///
    /// - Returns: Sum of square distances of samples to their closest cluster center
    public func score() -> Tensor<Float> {
        return self.inertia
    }
     
}