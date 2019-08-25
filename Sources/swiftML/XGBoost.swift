import Foundation
import TensorFlow
import Python

public class XGBoost{
    
    /// Importing xgboost and pyplot python modules
    let xgb = Python.import("xgboost")
    let plt = Python.import("matplotlib.pyplot")
    
    /// Train data
    public var train: PythonObject?
    /// Test data
    public var test: PythonObject?
    /// Fitted model
    public var model: PythonObject?
    
    
    /// Sets train data
    /// - Parameters:
    ///   - trainData: tensor of training data
    ///   - trainLabel: tensor of training label
    /// - Returns: XGBoost
    public init (trainData: Tensor<Float>, trainLabel: Tensor<Float>) {
        loadArray(data: trainData.array, label: trainLabel.array, type: "train")
    }
    
    /// Sets train data
    /// - Parameters:
    ///   - trainDataPath: path ot libSVM/XGBoostBinary format
    /// - Returns: XGBoost
    public init (trainDataPath: String) {
        loadFileData(filePath: trainDataPath, type: "train")
    }
    
    
    /// Fits a model to training data according to parameters
    /// - Parameters:
    ///   - parameters: dictionary of paramters
    ///   - iterations: number of rounds to be trained
    /// - Returns: None
    public func boost(parameters: Dictionary<String, String>,
                      iterations: Int = 10
        ){
        let param = (PythonObject)(parameters)
        let rounds = (PythonObject)(iterations)
        self.model = self.xgb.train(param, self.train!, rounds)
    }
    
    public func crossValidate(parameters: Dictionary<String, String>,
                              metrics: String = "rmse",
                              earlyStopAt: Int = 10,
                              nFold: Int = 3,
                              seed: Int = 123,
                              iterations: Int = 50
        ){
        let param = (PythonObject)(parameters)
        let rounds = (PythonObject)(iterations)
        self.model = xgb.cv(dtrain: self.train!,
                            params: param,
                            nfold:nFold,
                            num_boost_round:rounds,
                            early_stopping_rounds:earlyStopAt,
                            metrics:metrics,
                            seed:seed)
    }
    
    /// Loads a model from a file in the local system
    /// - Parameters: from: path ot the model file
    /// - Returns: None
    public func loadModel(from: String){
        self.model = xgb.load_model(from)
    }
    
    /// Get predictions from fitted model on test data
    ///   - testData: tensor of test data
    ///   - testLabel: tensor of test label
    /// - Returns: tensor of predictions
    public func predict(testData: Tensor<Float>,
                        testLabel: Tensor<Float>) -> Tensor<Float>
    {
        loadArray(data: testData.array, label: testLabel.array, type: "test")
        let predictions = self.model!.predict(self.test!)
        let output = Tensor<Float>(numpy: predictions)!
        return output
    }
    
    /// Get predictions from fitted model on test data
    ///   - testFile: path ot libSVM/XGBoostBinary format
    /// - Returns: tensor of predictions
    public func predict(testFile: String) -> Tensor<Float>
    {
        loadFileData(filePath: testFile, type: "test")
        let predictions = self.model!.predict(self.test!)
        let output = Tensor<Float>(numpy: predictions)!
        return output
    }
    
    /// Save a model from a file in the local system
    /// - Parameters: at: path where to savw the model file
    /// - Returns: None
    public func saveModel(at: String){
        self.model!.dump_model(at)
    }
    
    /// Save a model from a file in the local system
    /// - Parameters:
    ///   - fileAt: path where to savw the model file
    ///   - fileAt: path where to savw the feature file
    /// - Returns: None
    public func saveModel(fileAt: String, featuresAt: String){
        self.model!.dump_model(fileAt, featuresAt)
    }
    
    
    /// Plot importance of booster
    /// - Parameters: None
    /// - Returns: None
    public func plotImportance(){
        xgb.plot_importance(self.model!)
    }
    
    /// Plot the trees
    /// - Parameters: numTrees: number of trees to be printed
    /// - Returns: None
    public func plotTree(numTrees: Int = 1){
        xgb.plot_tree(self.model!, num_trees:numTrees)
    }
    
    
    /// Loads data from a file in the local system
    /// - Parameters:
    ///   - filePath: path ot the data file
    ///   - type: test or train
    /// - Returns: None
    public func loadFileData(filePath: String, type: String){
        if type == "train"{
            self.train = xgb.DMatrix(filePath)
        } else {
            self.test = xgb.DMatrix(filePath)
        }
    }
    
    /// Loads data from array
    /// - Parameters:
    ///   - data: array of data
    ///   - label: array of label
    ///   - type: test or train
    /// - Returns: None
    public func loadArray(data: ShapedArray<Float>,
                          label: ShapedArray<Float>,
                          type: String){
        let dataP = data.makeNumpyArray()
        let labelP = label.makeNumpyArray()
        if type == "train"{
            self.train = xgb.DMatrix(dataP, label: labelP)
        } else {
            self.test = xgb.DMatrix(dataP, label: labelP)
        }
    }
    
}
