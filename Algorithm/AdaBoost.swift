public class AdaBoost {
    public var stumps:[DecisionTree]
    public var weights:[Float]
    public var alphas:[Float]
    public var errors:[Float]
    public var iterations:Int
    public var data:[[String]]
    public var K:Float
    public var target:Int
    
    /*
     initializer for AdaBoost Classifier
     Parameters:
            Data -> data with labels
            Target -> column number of the labels
            Till -> number of stumps to be generated
            classes -> number of classes
    */
    public init (data: [[String]], target : Int, till : Int, classes: Int){
        self.data = data
        self.target = target
        let initialWeight = Float(1)/(Float)(data.count-1)
        self.weights = [Float](repeating: initialWeight, count: data.count-1)
        self.iterations = till
        self.stumps = [DecisionTree]()
        self.alphas = [Float]()
        self.errors = [Float]()
        self.K = (Float)(classes)
        
    }
    
    //method that creates the stumps and fills weights
    public func boost(){
        var stumpData = self.data
        for i in 1...self.iterations {
            let stump = DecisionTree(data: stumpData, target: self.target, maxDepth: 1, perform: "classification", using : "gini")
            self.stumps.append(stump)
            computeError(stump: stump)
            stumpData = getWeightSampledData(from : stumpData)
        }
    }
    
    //Method that computes Error for a given stump
    public func computeError(stump : DecisionTree) {
        let examples = deleteColumn(data: self.data, column: self.target)
        let head = self.data[0]
        var error : Float = 0.0
        let denominator : Float = self.weights.reduce(0, +)
        var correctIndices = [Int]()
        var incorrectIndices = [Int]()
        
        for i in 1 ... examples.count-1 {
            var example = [[String]]()
            example.append(head)
            example.append(examples[i])
            
            let stumpClassification = stump.classify(example:example)

            if stumpClassification == self.data[i][self.target] {
                correctIndices.append(i)
            } else {
                error += self.weights[i-1]
                incorrectIndices.append(i)
            }
            
        }
        
        let alpha = (log10((1-error)/error)+log10(self.K - 1))
        self.alphas.append(alpha)
        
        for i in correctIndices {
            self.weights[i-1] = self.weights[i-1]/(self.K*(1-error))
        }
        
        for j in incorrectIndices {
            self.weights[j-1] = self.weights[j-1]/(self.K*error)
        }
        
    }
    
    //method that returns new dataset with wieghted sampling based on previous errors
    public func getWeightSampledData(from : [[String]]) -> [[String]]{
        var sampledData = [[String]]()
        let head = self.data[0]
        sampledData.append(head)
        
        for i in 1...data.count-1 {
            let index = Float.random(in: 0 ..< 1)
            var chosenExample : Int = 1
            var currentWeight : Float = self.weights[0]
            
            while currentWeight < index {
                if(chosenExample > data.count-1){
                    break;
                }
                currentWeight += self.weights[chosenExample]
                chosenExample += 1
            }
            
            sampledData.append(from[chosenExample])
        }
        
        return sampledData
        
    }
    
    //function that classfies as example by getting all stump classification and weighing them
    public func classify(this: [[String]]) -> String{
        let result = Feature(data: self.data, column : self.target)
        var resultProbabilities = [Float]()
        var resultValues = [String]()
        
        for value in result.values {
            var probability : Float = 0.0
            resultValues.append(value.name)

            for i in 0...self.stumps.count-1 {
                let stump = self.stumps[i]        
                let stumpClassification = stump.classify(example:this)
                
                if stumpClassification == value.name {
                    probability += self.alphas[i]
                } 
            }
            resultProbabilities.append(probability)
        }

        let highestProbabilityIndex = resultProbabilities.firstIndex(of: resultProbabilities.max()!)!
        let classification = resultValues[highestProbabilityIndex]
        
        return classification
    }
    
}
