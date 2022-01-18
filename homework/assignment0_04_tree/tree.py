import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005
    if y.shape[0] == 0:
        return 0

    class_pr = np.mean(y, axis=0)
    
    return -np.sum(class_pr * np.log(class_pr + EPS))
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    return 0 if y.shape[0] == 0 else 1 - np.sum(np.mean(y, axis=0) ** 2)
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    return 0 if y.shape[0] == 0 else np.var(y)

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    return 0 if y.shape[0] == 0 else np.mean(np.abs(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, value=0, proba=0, is_leaf=False):
        self.feature_index = feature_index
        self.threshold = threshold
        self.proba = proba
        self.value = value
        self.left_child = None
        self.right_child = None
        self.is_leaf = is_leaf
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the provided subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the provided subset where selected feature x^j >= threshold
        """
        left_node_mask = X_subset[:,feature_index] < threshold

        X_left, y_left = X_subset[left_node_mask], y_subset[left_node_mask]
        X_right, y_right = X_subset[~left_node_mask], y_subset[~left_node_mask]
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        left_node_mask = X_subset[:,feature_index] < threshold
        return y_subset[left_node_mask], y_subset[~left_node_mask]

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # for every feature we greedily trying to find a threshold 
        # trying to minimize weighted average of entopies so that 
        # uncertanty in children would be lowest possible
        best_feature_ind, best_threshold, best_criterion_value = 0, 0.0, np.inf
        n_features = X_subset.shape[1]
        for feature_index in range(n_features):
            for cur_threshold in np.unique(X_subset[:, feature_index]):
                y_left, y_right = self.make_split_only_y(feature_index, cur_threshold, X_subset, y_subset)
                new_value = len(y_left) * self.criterion(y_left) + len(y_right) * self.criterion(y_right)
                
                if new_value < best_criterion_value:
                    best_feature_ind, best_threshold = feature_index, cur_threshold
                    best_criterion_value = new_value
                    
        return best_feature_ind, best_threshold
    
    def create_leaf(self, y_subset):
        if self.classification:
            return Node(
                0, 0, value=np.argmax(np.sum(y_subset, axis=0)), 
                proba=np.mean(y_subset, axis=0), 
                is_leaf=True)
        else:
            value = np.median(y_subset) if self.criterion_name == 'mad_median' else np.mean(y_subset)
            return Node(0, 0, value=value, is_leaf=True)
        return None
    
    def make_tree(self, X_subset, y_subset, current_depth):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        n_objects = X_subset.shape[0]
        
        if self.classification and self.criterion(y_subset) < 1e-6: # if all subset is one class
            return self.create_leaf(y_subset)
        
        if current_depth >= self.max_depth or n_objects < self.min_samples_split:
            return self.create_leaf(y_subset)
        
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)

        (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
        
        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            return self.create_leaf(y_subset)
        
        new_node = Node(feature_index, threshold)
        new_node.left_child = self.make_tree(X_left, y_left, current_depth + 1)
        new_node.right_child = self.make_tree(X_right, y_right, current_depth + 1)
        
        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y, 0)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        y_predicted = np.zeros(X.shape[0])
        for i, new_obj in enumerate(X):
            # traverse the tree 
            cur_node = self.root
            while not cur_node.is_leaf:
                cur_node = cur_node.left_child if new_obj[cur_node.feature_index] < cur_node.threshold else cur_node.right_child
            y_predicted[i] = cur_node.value
        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        y_predicted_probs = np.ndarray((X.shape[0], self.n_classes))
        for i, new_obj in enumerate(X):
            # traverse the tree 
            cur_node = self.root
            while not cur_node.is_leaf:
                cur_node = cur_node.left_child if new_obj[cur_node.feature_index] < cur_node.threshold else cur_node.right_child
            y_predicted_probs[i] = cur_node.proba
        
        return y_predicted_probs
