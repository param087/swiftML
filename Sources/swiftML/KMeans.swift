import TensorFlow

/// K-Means Clustering
public class KMeans {
    /// Method for initialization.
    var initializer: String
    /// Seed for initializing the pseudo-random number generator.
    var seed: Int64
    /// Maximum number of iterations of the k-means algorithm to run.
    var maximumIterationCount: Int
    /// The number of clusters to form as well as the number of centroids to generate.
    var clusterCount: Int
    /// Cluster centers which data is assigned to.
    public var centroids: Tensor<Float>
    /// Inertia is the sum of square distances of samples to their closest cluster center.
    public var inertia: Tensor<Float>
    /// Predicted cluster for training data.
    public var labels: Tensor<Int32>

  
    /// Creates a logistic regression model.
    ///
    /// - Parameters
    ///   - clusterCount: The number of clusters to form as well as the number of centroids to
    ///     generate, default to `2`.
    ///   - maximumIterationCount: Maximum number of iterations of the k-means algorithm to run,
    ///     default to `300`.
    ///   - initializer: Select the initialization method for centroids. `kmean++`, `random`
    ///     methods for initialization, default to `kmean++`.
    ///   - seed: Used to initialize a pseudo-random number generator, default to `0`.
    public init(
        clusterCount: Int = 2,
        maximumIterationCount: Int = 300,
        initializer: String = "kmean++",
        seed: Int64 = 0
    ) {

        precondition(clusterCount > 1, "Clusters count must be greater than one.")
        precondition(maximumIterationCount >= 0, "Maximum iteration count must be non-negative.")
        precondition(initializer == "kmean++" || initializer == "random",
            "Intialializer must be `keman++` or `random`.")
        
        self.clusterCount = clusterCount
        self.maximumIterationCount = maximumIterationCount
        self.initializer = initializer
        self.seed = seed
        self.centroids = Tensor<Float>([0.0])
        self.inertia = Tensor<Float>(0)
        self.labels = Tensor<Int32>([0])
    }
  
    /// Return the index of centroid having minimum euclidean distance with data.
    ///
    /// - Parameters
    ///   - centroids: Centroids with shape `[1, feature count]`.
    ///   - data: Data tensor of shape `[sample count, feature count]`.
    /// - Returns: Index of minimum euclidean distance centroid.
    internal func nearest(centroids: Tensor<Float>, data: Tensor<Float>) -> Tensor<Int32> {
        var distances = Tensor<Float>(zeros: [centroids.shape[0], 1])
        for i in 0..<centroids.shape[0] {
            distances[i][0] = euclideanDistance(centroids[i], data)
        }
        return distances.argmin()
    }

    /// Heuristic Initialization of centroids.
    ///
    /// Reference: ["k-means++: The Advantages of Careful Seeding"](
    /// http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)
    ///
    /// - Parameter data: Data with shape `[sample count, feature count]`.
    internal func kmeanPlusPlus(_ data: Tensor<Float>) {
        var distance = Tensor<Float>(zeros: [data.shape[0], 1])
        
        // Choose first center uniformly at random from among the data points.
        let index: Int = Int.random(in: 0..<data.shape[0])
        self.centroids[0] = data[index]

        // For each data point `x`, compute `D(x)`, the distance between `x`and the nearest center
        // that has already been chosen.
        // Choose one new data point at random as a new center, using a weighted probability
        // distribution where a point `x` is chosen with probability proportional to `D(x)`
        for i in 1..<self.clusterCount {

            for dataIndex in 0..<data.shape[0] {
                var d = Tensor<Float>(zeros: [i, 1])
                for centerIndex in 0..<i {
                    d[centerIndex][0] = 
                        euclideanDistance(self.centroids[centerIndex], data[dataIndex])
                }
                distance[dataIndex][0] = d.min()
            }

            let probability = distance / distance.sum()
            let cumsum = Raw.cumsum(probability.flattened(), axis: Tensor<Int32>(-1))
            let threshold = Tensor<Float>(Float.random(in: 0...1))

            var index: Int = 0

            for j in 0..<cumsum.shape[0] {
                if cumsum[j] >= threshold {
                    break
                } else {
                    index = index + 1
                }
            }
            self.centroids[i] = data[index]
        }
    }
    
    /// Random Initialization of centroids.
    ///
    /// - Parameter data: Data with shape `[sample count, feature count]`.
    internal func randomInitializer(_ data: Tensor<Float>) {
        // shuffle the input data.
        let shuffled = Raw.randomShuffle(value: data, seed: self.seed)
        for i in 0..<clusterCount {
            self.centroids[i] = shuffled[i]
        }
    }
    
    /// Compute k-means clustering.
    ///
    /// - Parameter data: Input data with shape `[sample count, feature count]`.
    public func fit(data: Tensor<Float>) {
        precondition(data.shape[0] > 0, "Data must be non-empty.")

        // reshape centroid of required shape based on input data and number of clusters.
        self.centroids = Tensor<Float>(zeros: [clusterCount, data.shape[1]])
        self.labels = Tensor<Int32>(zeros: [data.shape[0], 1])
        
        if self.initializer == "kmean++" {
            self.kmeanPlusPlus(data)
        } else if self.initializer == "random" {
            self.randomInitializer(data)
        }
        
        var oldCentroids = centroids

        for _ in 0..<self.maximumIterationCount {
            
            var indicesArray = [[Tensor<Int32>]]()
            
            for i in 0..<data.shape[0] {
                self.labels[i][0] = self.nearest(centroids: self.centroids, data: data[i])
            }
            
            for i in 0..<self.clusterCount {
                var labelIndexArray = [Tensor<Int32>]()
                for j in 0..<self.labels.shape[0] {
                    if self.labels[j].scalarized() == i {
                        let labelIndex = Tensor<Int32>(Int32(j))
                        labelIndexArray.append(labelIndex)
                    }
                }
                indicesArray.append(labelIndexArray)
            }

            // Update the cluster centroids based on average of element in cluster. 
            for i in 0..<indicesArray.count {
                var temp = Tensor<Float>(zeros: [1, data.shape[1]])
                for j in 0..<indicesArray[i].count {
                    let index: Int = Int(indicesArray[i][j].scalarized())
                    temp = temp + data[index]
                }
                
                temp = temp * Tensor<Float>((Float(1) / Float(indicesArray[i].count)))
                self.centroids[i] = temp[0]
            }

            if oldCentroids == centroids {
                break
            } else {
                oldCentroids = centroids
            }

        }
        
        // sum of square distances from the closest cluster.
        for i in 0..<data.shape[0] {
            self.inertia = self.inertia + 
                pow((self.centroids[Int(self.labels[i].scalarized())] - data[i]), 2).sum()
        }       
    }
  
    /// Returns the prediced cluster labels.
    ///
    /// - Parameter data: Input data with shape `[sample count, feature count]`.
    /// - Returns: Predicted prediction for input data.
    public func prediction(for data: Tensor<Float>) -> Tensor<Int32> {
        precondition(data.shape[0] > 0, "Data must be non-empty.")

        var labels = Tensor<Int32>(zeros: [data.shape[0], 1])
        for i in 0..<data.shape[0] {
            labels[i][0] = self.nearest(centroids: self.centroids, data: data[i])
        }
        return labels       
    }
  
    /// Returns fit and prediced cluster labels.
    ///
    /// - Parameter data: Input data with shape `[sample count, feature count]`.
    /// - Returns: Predicted prediction for input data.
    public func fitAndPrediction(for data: Tensor<Float>) -> Tensor<Int32> {
        precondition(data.shape[0] > 0, "Data must be non-empty.")
        
        self.fit(data: data)
        return self.prediction(for: data)
    }
    
    /// Returns Transform input to a cluster-distance space.
    ///
    /// - Parameter data: Input data with shape `[sample count, feature count]`.
    /// - Returns: Transformed input to a cluster-distance space.
    public func transformation(for data: Tensor<Float>) -> Tensor<Float> {
        precondition(data.shape[0] > 0, "Data must be non-empty.")

        var transMat = Tensor<Float>(zeros:[data.shape[0], self.clusterCount])
        
        for i in 0..<data.shape[0] {
            for j in 0..<self.clusterCount {
                transMat[i][j] = euclideanDistance(data[i], self.centroids[j])
            }
        }
        return transMat
    }

    /// Returns fit and Transform input to a cluster-distance space.
    ///
    /// - Parameter data: Input data with shape `[sample count, feature count]`.
    /// - Returns: Transformed data to a cluster-distance space.
    public func fitAndTransformation(for data: Tensor<Float>) -> Tensor<Float> {
        precondition(data.shape[0] > 0, "Data must be non-empty.")

        self.fit(data: data)
        return self.transformation(for: data)
    }
    
    /// Returns the sum of square distances of samples to their closest cluster center.
    ///
    /// - Returns: Sum of square distances of samples to their closest cluster center.
    public func score() -> Tensor<Float> {
        return self.inertia
    }

}