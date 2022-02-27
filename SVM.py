import re
import numpy as np
from collections import Counter, OrderedDict
from sklearn.metrics.pairwise import euclidean_distances  # , polynomial_kernel


# Superclass
class Classifier:
    def __init__(self):
        self.train_raw = None
        self.test_raw = None
        self.model_lines = None
        self.n_classes = None
        self.n_docs = None
        # self.n_docs_in_class = None
        self.feature_set = set()  # Set
        self.n_features = None  # int
        self.n_docs = None
        self.labels = set()
        self.vocab2idx = None
        self.class2idx = None
        self.idx2vocab = None
        self.idx2class = None
        self.neighbors = None
        self.y_hat_train = None
        self.y_hat_train_probs = None
        self.y_hat_test = None
        self.y_hat_test_probs = None

    @staticmethod
    def create_array(rows, columns):
        array_out = np.zeros((rows, columns))
        return array_out

    @staticmethod
    def prep_documents(data, type):
        data_clean = [re.sub('\n', '', x) for x in data]
        data_clean = [x for x in data_clean if x]
        data_clean = [re.split(r"\s+", x) for x in data_clean]

        # Split X and y
        y_str = [x[0] for x in data_clean]
        X_str = [x[1:] for x in data_clean]

        if type == 'count':
            X_str = [[sl for sl in l if sl] for l in X_str]
            X_str = [[tuple(re.split(r":", sl)) for sl in l] for l in X_str]
            X_str = [dict(l) for l in X_str]
            X_str = [dict((k, int(v)) for k, v in subdict.items()) for subdict in X_str]
        else:
            X_str = [[re.split(r":", sl)[0] for sl in l] for l in X_str]
            X_str = [[sl for sl in l if sl] for l in X_str]

        return X_str, y_str

    def confusion_matrix(self, y_actual, y_predicted):
        conf_matrix = np.zeros((self.n_classes, self.n_classes))
        # print(conf_matrix)
        for actual, pred in zip(y_actual, y_predicted):
            # print(str(actual) +' '+ str(pred))
            conf_matrix[actual, pred] += 1
        return conf_matrix

    def get_acc(self, y_actual, y_predicted):
        actual = np.array(y_actual)
        pred = np.array(y_predicted)
        correct = (actual == pred)
        accuracy = correct.sum() / correct.size
        return accuracy

    def classification_report(self, test_only=None):
        output_lines = []
        test_matrix = self.confusion_matrix(self.y_test, self.y_hat_test)
        test_acc = self.get_acc(self.y_test, self.y_hat_test)

        class_labels = [str(label) for label in list(self.labels)]  # list(self.labels)

        class_labels_join = ' '.join(class_labels)
        class_labels_join = "\t\t" + class_labels_join

        if self.y_hat_train is not None:
            train_header_line = ['Confusion matrix for the training data:',
                                 'row is the truth, column is the system output',
                                 '\n']
            output_lines.append(train_header_line)

            train_matrix = self.confusion_matrix(self.y_train, self.y_hat_train)

            train_acc = self.get_acc(self.y_train, self.y_hat_train)

            output_lines.append(class_labels_join)

            for key, value in self.idx2class.items():
                matrix_counts = train_matrix[key, :].tolist()
                matrix_counts = [str(int(x)) for x in matrix_counts]
                matrix_counts = ' '.join(matrix_counts)
                matrix_line = str(value) + ' ' + matrix_counts
                output_lines.append(matrix_line)

            output_lines.append('\n')
            output_lines.append("Training accuracy=" + str(train_acc))
            output_lines.append('\n')

        second_title = ['Confusion matrix for the test data:', 'row is the truth, column is the system output',
                        '\n']
        output_lines += second_title
        output_lines.append(class_labels_join)

        # for key, value in self.idx2class.items():
        for value in class_labels:
            key = int(value)
            matrix_counts = test_matrix[key, :].tolist()
            matrix_counts = [str(int(x)) for x in matrix_counts]
            matrix_counts = ' '.join(matrix_counts)
            matrix_line = str(value) + ' ' + matrix_counts
            output_lines.append(matrix_line)

        output_lines.append('\n')
        output_lines.append("Test accuracy=" + str(test_acc))

        for line in output_lines:
            print(line)

    def save_sys_output(self, sys_output_dir, test_only=None):
        """
        Write predictions to file
        """
        output_lines = []
        for inst_idx, actual_label in enumerate(self.y_test):
            pred_label = self.y_hat_test[inst_idx]
            fx = self.fx[inst_idx]
            line = str(actual_label) + " " + str(pred_label) + " " + str(fx)
            output_lines.append(line)

        with open(sys_output_dir, 'w', encoding='utf8') as f:
            f.writelines("%s\n" % line for line in output_lines)


# Sublcass
class SVMClassifier(Classifier):
    def __init__(self):
        self.X_train = None  # np array
        self.y_train = None  # np array
        self.y_train_M = None
        self.X_test = None  # np array
        self.y_test = None  # np array
        self.kernel = None
        self.total_sv = None
        self.nr_sv = None
        self.rho = None
        self.support_vectors = None
        self.sv_weights = None
        self.gamma = None
        self.coef = None
        self.degree = None
        self.fx = []
        super().__init__()

    def load_model(self, model_lines):
        data_clean = [re.sub('\n', '', x) for x in model_lines]
        sv_idx = int(data_clean.index('SV'))
        header = data_clean[: sv_idx]
        header = [line.split() for line in header]
        sv_raw = data_clean[sv_idx + 1:]

        self.kernel = [x[1] for x in header if x[0] == 'kernel_type'][0]
        self.n_classes = [int(x[1]) for x in header if x[0] == 'nr_class'][0]
        self.rho = [float(x[1]) for x in header if x[0] == 'rho'][0]
        self.total_sv = [int(x[1]) for x in header if x[0] == 'total_sv'][0]
        self.nr_sv = [x[1:] for x in header if x[0] == 'nr_sv']

        labels = [x[1:] for x in header if x[0] == 'label']
        self.labels = set([int(label) for label in labels[0]])

        if self.kernel != 'linear':
            self.gamma = [x[1:] for x in header if x[0] == 'gamma'][0][0]
            self.gamma = float(self.gamma)

            if self.kernel != 'rbf':
                self.coef = [x[1:] for x in header if x[0] == 'coef0'][0][0]
                self.coef = float(self.coef)

            if self.kernel == 'polynomial':
                self.degree = [x[1:] for x in header if x[0] == 'degree'][0][0]
                self.degree = int(self.degree)

        sv_split = [line.split() for line in sv_raw]
        self.sv_weights = np.array([float(line[0]) for line in sv_split])
        features = [line[1:] for line in sv_split]
        sv_features = [[int(item.split(":")[0]) for item in sl] for sl in features]
        self.feature_set = set([feature for sl in sv_features for feature in sl])
        self.n_features = max(self.feature_set)

        self.support_vectors = self.create_array(self.total_sv, self.n_features + 1)

        for sv_idx, sv_feature_list in enumerate(sv_features):
            for feat_idx in sv_feature_list:
                self.support_vectors[sv_idx, feat_idx] = 1

    def process_test(self):
        X_ts, y_ts = self.prep_documents(self.test_raw, type='binary')

        # Set class information
        if self.class2idx is not None:
            self.y_test = np.array([self.class2idx[c] for c in y_ts if c in self.class2idx])
        else:
            self.y_test = np.array([int(label) for label in y_ts])

        # Set number of docs
        n_test_docs = len(y_ts)

        X_array = [[int(feature) for feature in doc if int(feature) in self.feature_set] for doc in X_ts]

        self.X_test = self.create_array(n_test_docs, self.n_features + 1)

        for doc_idx, doc in enumerate(X_array):
            for word in doc:
                self.X_test[doc_idx, word] = 1

    def _linear_kernel_fn(self, ts_instance):
        k = np.multiply(self.support_vectors, ts_instance)
        weighted = k * self.sv_weights[:, None]
        fx = np.sum(weighted, axis=None)
        fx = fx - self.rho

        if fx > 0.0:
            pred = 0
        else:
            pred = 1

        return pred, fx

    def _polynomial_kernel_fn(self, ts_instance):
        ts_instance = np.array([ts_instance])
        k = np.dot(self.support_vectors, ts_instance.T)
        k = k * self.gamma
        k = k + self.coef
        k = k ** self.degree

        weighted = k * self.sv_weights[:, None]
        fx = np.sum(weighted, axis=None)
        fx = fx - self.rho

        if fx > 0.0:
            pred = 0
        else:
            pred = 1

        return pred, fx

    def _rbf_kernel_fn(self, ts_instance):
        ts_instance = np.array([ts_instance])
        k = euclidean_distances(self.support_vectors, ts_instance, squared=True)
        k *= -self.gamma
        k = np.exp(k)
        weighted = k * self.sv_weights[:, None]
        fx = np.sum(weighted, axis=None)
        fx = fx - self.rho

        if fx > 0.0:
            pred = 0
        else:
            pred = 1

        return pred, fx

    def _sigmoid_kernel_fn(self, ts_instance):
        ts_instance = np.array([ts_instance])
        k = np.dot(self.support_vectors, ts_instance.T)
        k = k * self.gamma
        k = k + self.coef
        k = np.tanh(k)
        weighted = k * self.sv_weights[:, None]
        fx = np.sum(weighted, axis=None)
        fx = fx - self.rho

        if fx > 0.0:
            pred = 0
        else:
            pred = 1

        return pred, fx

    def predict(self, instances, save):
        y_pred = np.zeros(np.shape(instances)[0]).astype(int)
        y_fx = []

        if self.kernel == 'linear':
            for doc_idx, doc in enumerate(instances):
                c_pred, c_fx = self._linear_kernel_fn(instances[doc_idx, :])
                y_pred[doc_idx] = c_pred
                y_fx.append(c_fx)
        elif self.kernel == 'polynomial':
            for doc_idx, doc in enumerate(instances):
                c_pred, c_fx = self._polynomial_kernel_fn(instances[doc_idx, :])
                y_pred[doc_idx] = c_pred
                y_fx.append(c_fx)
        elif self.kernel == 'rbf':
            for doc_idx, doc in enumerate(instances):
                c_pred, c_fx = self._rbf_kernel_fn(instances[doc_idx, :])
                y_pred[doc_idx] = c_pred
                y_fx.append(c_fx)
        elif self.kernel == 'sigmoid':
            for doc_idx, doc in enumerate(instances):
                c_pred, c_fx = self._sigmoid_kernel_fn(instances[doc_idx, :])
                y_pred[doc_idx] = c_pred
                y_fx.append(c_fx)

        if save == 'test':
            self.y_hat_test = y_pred
            self.fx = y_fx
        elif save == 'train':
            self.y_hat_train = y_pred

        return y_pred, y_fx
