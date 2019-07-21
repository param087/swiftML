public class RandomForest{
    
    public var forest : [RandomTree]
    public var fullData : [[String]]
    public var target : Int
    public var nTrees : Int
    public var nFeatures : Int
    public var depth : Int
    public var perform : String
    public var using : String
    
    /*
     Init for Random Forest
     Parameters:
        data -> data with labels
        target -> label column number
        perform -> regression or classification
        using -> giniIndex or infoGain
        nTrees -> number of Trees
    */
    public init (data: [[String]], target : Int, perform : String, using: String, nTrees: Int, nFeatures: Int, depth: Int) {
        self.fullData = data
        self.target = target
        self.nTrees = nTrees
        self.nFeatures = nFeatures
        self.depth = depth
        self.forest = [RandomTree]()
        self.perform = perform
        self.using = using
    }
    
    //method that makes the tree using bootstrapped data
    public func make() {
        var bootData : [[String]]
        var outOfBootData: [[String]]
        
        for i in 0...nTrees {
            let data = splitData(from: self.fullData)
            bootData = data.0
            outOfBootData = data.1
            let tree = RandomTree(data: bootData, target: self.target, perform: self.perform, using: self.using, with: self.nFeatures)
            self.forest.append(tree)
        }
    }
    
    //method that returns bootstrapped and out of bootData
    public func splitData(from: [[String]]) -> ([[String]],[[String]]) {
        var selectedIndices = [Int]()
        let head = from[0]
        var bootstrappedData = [[String]]()
        bootstrappedData.append(head)
        
        for i in 0...from.count-1 {
            let index = Int.random(in: 1 ... from.count-1)
            bootstrappedData.append(from[index])
            selectedIndices.append(index)
        }
        
        var testData = [[String]]()
        testData.append(head)
        for i in 1...from.count-1 {
            if !selectedIndices.contains(i){
                testData.append(from[i])
            }
        }
        
        return (bootstrappedData, testData)
    }
    
    
    //method that predicts/classifies an example using the Forest
    public func predict(example: [[String]]) -> String{
        if self.perform == "regression"{
            var prediction : Float = 0.0
            for tree in self.forest {
                prediction += (Float)(tree.classify(example:example))!
            }
            return (String)(prediction/(Float)(self.forest.count))
            
        } else {
            let target = Feature(data : self.fullData, column : self.target)
            var classes = Dictionary<String, Int>()
            
            for value in target.values {
                classes[value.name] = 0
            }
            
            for tree in self.forest {
                var classification = tree.classify(example:example)
                classes[classification] = classes[classification]! + 1
            }

            let maxClass = classes.max { a, b in a.value < b.value }
            
            return maxClass!.key
        }
    }
    
}
