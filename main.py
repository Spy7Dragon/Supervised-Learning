from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

columns = [
    "annual_inc",
    "collections_12_mths_ex_med",
    "delinq_amnt",
    "delinq_2yrs",
    "dti",
    # "earliest_cr_line", #possibly remove because age doesn't matter
    # "emp_length", #possibly remove because age doesn't matter
    "fico_range_high",
    "fico_range_low",
    "home_ownership",
    "inq_last_6mths",
    "installment",
    "int_rate",
    "verification_status",
    "loan_amnt",
    "mths_since_last_major_derog",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "purpose",
    "revol_bal",
    "revol_util",
    "sub_grade",
    "term",
    "total_acc",
    "loan_status"
]

encodings = {
        "term" : { "36 months": 0, "60 months" : 1},
        "sub_grade" : {"A1": 0, "A2" : 1, "A3" : 2, "A4" : 3, "A5" : 4,
                       "B1": 5, "B2": 6, "B3": 7, "B4": 8, "B5": 9,
                       "C1": 10, "C2": 11, "C3": 12, "C4": 13, "C5": 14,
                       "D1": 15, "D2": 16, "D3": 17, "D4": 18, "D5": 19,
                       "E1": 20, "E2": 21, "E3": 22, "E4": 23, "E5": 24,
                       "F1": 25, "F2": 26, "F3": 27, "F4": 28, "F5": 29,
                       "G1": 30, "G2": 31, "G3": 32, "G4": 33, "G5": 34},
        "home_ownership": { "OWN": 0, "MORTGAGE": 1, "RENT": 2, "OTHER": 3, "NONE": 4},
        "verification_status": {"Not Verified": 0, "Verified": 1, "Source Verified": 2},
        "loan_status": {"Charged Off": 0, "Fully Paid": 1},
        "purpose": {"debt_consolidation": 3,
                    "credit_card": 4,
                    "other": 13,
                    "home_improvement": 8,
                    "major_purchase": 12,
                    "small_business": 0,
                    "car": 7,
                    "wedding": 10,
                    "medical": 6,
                    "moving": 5,
                    "house": 9,
                    "educational": 2,
                    "vacation": 11,
                    "renewable_energy": 1},
        "int_rate": {"%" : ""},
        "revol_util": {"%" : ""}
    }


def perform_training_size(model, name):
    print("Perform training size for " + name)
    intervals = 20
    percentage = 1.0 / intervals
    training_size = training_classes.shape[0]
    section_scores = []
    for i in range(1, intervals + 1):
        training_percentage = percentage * i
        training_section_size = int(training_percentage * training_size)
        training_features_section = training_features[:training_section_size]
        training_classes_section = training_classes[:training_section_size]
        model.fit(training_features_section, training_classes_section)
        predicted_training_classes_section = model.predict(training_features)
        predicted_test_classes_section = model.predict(test_features)
        training_score = accuracy_score(training_classes, predicted_training_classes_section)
        training_error = 1.0 - training_score
        test_score = accuracy_score(test_classes, predicted_test_classes_section)
        test_error = 1.0 - test_score
        section_scores.append([training_section_size, training_error, test_error])

    plot_frame = pd.DataFrame(section_scores, columns=['Training Size', 'Training Error', 'Test Error'])
    graph = plot_frame.plot(x='Training Size', y=['Training Error', 'Test Error'], title=name + '-Training Size')
    graph.set_xlabel('Training Size')
    graph.set_ylabel('Error')
    plt.ylim(0.0, 1.0)
    plt.savefig('graphs/' + name + '-Training Size.png')


def perform_iterations(model, name):
    if not hasattr(model, 'max_iter'):
        return
    print("Perform iterations for " + name)
    intervals = 20
    interval_size = 100
    section_scores = []
    for i in range(1, intervals + 1):
        iterations = i * interval_size
        model.max_iter = iterations
        model.fit(training_features, training_classes)
        predicted_training_classes_section = model.predict(training_features)
        predicted_test_classes_section = model.predict(test_features)
        training_score = accuracy_score(training_classes, predicted_training_classes_section)
        training_error = 1.0 - training_score
        test_score = accuracy_score(test_classes, predicted_test_classes_section)
        test_error = 1.0 - test_score
        section_scores.append([iterations, training_error, test_error])

    plot_frame = pd.DataFrame(section_scores, columns=['Iterations', 'Training Error', 'Test Error'])
    graph = plot_frame.plot(x='Iterations', y=['Training Error', 'Test Error'], title=name + '-Iterations')
    graph.set_xlabel('Iterations')
    graph.set_ylabel('Error')
    plt.ylim(0.0, 0.5)
    plt.savefig('graphs/' + name + '-Iterations.png')


def perform_decision_tree_analysis(model, name):
    print("Perform analysis for " + name)
    # max_depth
    intervals = 20
    interval_size = 2
    section_scores = []
    for i in range(1, intervals + 1):
        max_depth = int(i * interval_size)
        model.max_depth = max_depth
        model.fit(training_features, training_classes)
        predicted_training_classes_section = model.predict(training_features)
        predicted_test_classes_section = model.predict(test_features)
        training_score = accuracy_score(training_classes, predicted_training_classes_section)
        training_error = 1.0 - training_score
        test_score = accuracy_score(test_classes, predicted_test_classes_section)
        test_error = 1.0 - test_score
        section_scores.append([max_depth, training_error, test_error])

    plot_frame = pd.DataFrame(section_scores, columns=['Max Depth', 'Training Error', 'Test Error'])
    graph = plot_frame.plot(x='Max Depth', y=['Training Error', 'Test Error'], title=name + '-Max Depth')
    graph.set_xlabel('Max Depth')
    graph.set_ylabel('Error')
    plt.ylim(0.0, 0.5)
    plt.savefig('graphs/' + name + '-Max Depth.png')
    # min_samples_split
    # intervals = 20
    # interval_size = 0.05
    # section_scores = []
    # for i in range(1, intervals + 1):
    #     min_samples_split = float(i * interval_size)
    #     classifier.min_samples_split = min_samples_split
    #     classifier.fit(training_features, training_classes)
    #     predicted_training_classes_section = classifier.predict(training_features)
    #     predicted_test_classes_section = classifier.predict(test_features)
    #     training_score = accuracy_score(training_classes, predicted_training_classes_section)
    #     training_error = 1.0 - training_score
    #     test_score = accuracy_score(test_classes, predicted_test_classes_section)
    #     test_error = 1.0 - test_score
    #     section_scores.append([min_samples_split, training_error, test_error])
    #
    # ind_var_name = "Min Samples Split"
    # plot_frame = pd.DataFrame(section_scores, columns=[ind_var_name, 'Training Error', 'Test Error'])
    # graph = plot_frame.plot(x=ind_var_name, y=['Training Error', 'Test Error'], title=model_name + '-' + ind_var_name)
    # graph.set_xlabel(ind_var_name)
    # graph.set_ylabel('Error')
    # plt.ylim(0.0, 0.5)
    # plt.savefig('graphs/' + model_name + '-' + ind_var_name + '.png')
    # criterion
    # criteria = ['gini', 'entropy']
    # section_scores = []
    # for i in range(0, len(criteria)):
    #     criterion = criteria[0]
    #     classifier.criterion = criterion
    #     classifier.fit(training_features, training_classes)
    #     predicted_training_classes_section = classifier.predict(training_features)
    #     predicted_test_classes_section = classifier.predict(test_features)
    #     training_score = accuracy_score(training_classes, predicted_training_classes_section)
    #     training_error = 1.0 - training_score
    #     test_score = accuracy_score(test_classes, predicted_test_classes_section)
    #     test_error = 1.0 - test_score
    #     section_scores.append([criterion, training_error, test_error])
    #
    # ind_var_name = "Criterion"
    # plot_frame = pd.DataFrame(section_scores, columns=[ind_var_name, 'Training Error', 'Test Error'])
    # graph = plot_frame.plot(x=ind_var_name, y=['Training Error', 'Test Error'], title=model_name + '-' + ind_var_name)
    # graph.set_xlabel(ind_var_name)
    # graph.set_ylabel('Error')
    # plt.ylim(0.0, 0.5)
    # plt.savefig('graphs/' + model_name + '-' + ind_var_name + '.png')


def perform_neural_net_analysis(model, name):
    print("Perform analysis for " + name)
    # hidden_layer_sizes
    intervals = 20
    interval_size = 20.0 / intervals
    section_scores = []
    for i in range(1, intervals + 1):
        hidden_layer_sizes = int(i * interval_size)
        model.hidden_layer_sizes = hidden_layer_sizes
        model.fit(training_features, training_classes)
        predicted_training_classes_section = model.predict(training_features)
        predicted_test_classes_section = model.predict(test_features)
        training_score = accuracy_score(training_classes, predicted_training_classes_section)
        training_error = 1.0 - training_score
        test_score = accuracy_score(test_classes, predicted_test_classes_section)
        test_error = 1.0 - test_score
        section_scores.append([hidden_layer_sizes, training_error, test_error])

    ind_var_name = 'Hidden Layer Size'
    plot_frame = pd.DataFrame(section_scores, columns=[ind_var_name, 'Training Error', 'Test Error'])
    graph = plot_frame.plot(x=ind_var_name, y=['Training Error', 'Test Error'], title=name + '-' + ind_var_name)
    graph.set_xlabel(ind_var_name)
    graph.set_ylabel('Error')
    plt.ylim(0.0, 0.5)
    plt.savefig('graphs/' + name + '-' + ind_var_name + '.png')
    # activation
    # activations = ['identity', 'logistic', 'tanh', 'relu']
    # section_scores = []
    # for i in range(0, len(activations)):
    #     activation = activations[i]
    #     classifier.activation = activation
    #     classifier.fit(training_features, training_classes)
    #     predicted_training_classes_section = classifier.predict(training_features)
    #     predicted_test_classes_section = classifier.predict(test_features)
    #     training_score = accuracy_score(training_classes, predicted_training_classes_section)
    #     training_error = 1.0 - training_score
    #     test_score = accuracy_score(test_classes, predicted_test_classes_section)
    #     test_error = 1.0 - test_score
    #     section_scores.append([activation, training_error, test_error])
    #
    # ind_var_name = 'Activation'
    # plot_frame = pd.DataFrame(section_scores, columns=[ind_var_name, 'Training Error', 'Test Error'])
    # graph = plot_frame.plot(x=ind_var_name, y=['Training Error', 'Test Error'], title=model_name + '-' + ind_var_name)
    # graph.set_xlabel(ind_var_name)
    # graph.set_ylabel('Error')
    # plt.ylim(0.0, 0.5)
    # plt.savefig('graphs/' + model_name + '-' + ind_var_name + '.png')
    # solver
    # solvers = ['lbfgs', 'sgd', 'adam']
    # section_scores = []
    # for i in range(0, len(solvers)):
    #     solver = solvers[i]
    #     classifier.solver = solver
    #     classifier.fit(training_features, training_classes)
    #     predicted_training_classes_section = classifier.predict(training_features)
    #     predicted_test_classes_section = classifier.predict(test_features)
    #     training_score = accuracy_score(training_classes, predicted_training_classes_section)
    #     training_error = 1.0 - training_score
    #     test_score = accuracy_score(test_classes, predicted_test_classes_section)
    #     test_error = 1.0 - test_score
    #     section_scores.append([solver, training_error, test_error])
    #
    # ind_var_name = 'Solver'
    # plot_frame = pd.DataFrame(section_scores, columns=[ind_var_name, 'Training Error', 'Test Error'])
    # graph = plot_frame.plot(x=ind_var_name, y=['Training Error', 'Test Error'], title=model_name + '-' + ind_var_name)
    # graph.set_xlabel(ind_var_name)
    # graph.set_ylabel('Error')
    # plt.ylim(0.0, 0.5)
    # plt.savefig('graphs/' + model_name + '-' + ind_var_name + '.png')
    # learning_rate
    # learning_rates = ['constant', 'invscaling', 'adaptive']
    # section_scores = []
    # for i in range(0, len(learning_rates)):
    #     learning_rate = learning_rates[i]
    #     classifier.learning_rate = learning_rate
    #     classifier.fit(training_features, training_classes)
    #     predicted_training_classes_section = classifier.predict(training_features)
    #     predicted_test_classes_section = classifier.predict(test_features)
    #     training_score = accuracy_score(training_classes, predicted_training_classes_section)
    #     training_error = 1.0 - training_score
    #     test_score = accuracy_score(test_classes, predicted_test_classes_section)
    #     test_error = 1.0 - test_score
    #     section_scores.append([learning_rate, training_error, test_error])
    #
    # ind_var_name = 'Learning Rate Schedule'
    # plot_frame = pd.DataFrame(section_scores, columns=[ind_var_name, 'Training Error', 'Test Error'])
    # graph = plot_frame.plot(x=ind_var_name, y=['Training Error', 'Test Error'], title=model_name + '-' + ind_var_name)
    # graph.set_xlabel(ind_var_name)
    # graph.set_ylabel('Error')
    # plt.ylim(0.0, 0.5)
    # plt.savefig('graphs/' + model_name + '-' + ind_var_name + '.png')
    # learning_rate_init
    # intervals = 20
    # interval_size = 0.00001
    # section_scores = []
    # for i in range(1, intervals + 1):
    #     learning_rate_init = i * interval_size
    #     classifier.learning_rate_init = learning_rate_init
    #     classifier.fit(training_features, training_classes)
    #     predicted_training_classes_section = classifier.predict(training_features)
    #     predicted_test_classes_section = classifier.predict(test_features)
    #     training_score = accuracy_score(training_classes, predicted_training_classes_section)
    #     training_error = 1.0 - training_score
    #     test_score = accuracy_score(test_classes, predicted_test_classes_section)
    #     test_error = 1.0 - test_score
    #     section_scores.append([learning_rate_init, training_error, test_error])
    #
    # ind_var_name = 'Learning Rate'
    # plot_frame = pd.DataFrame(section_scores, columns=[ind_var_name, 'Training Error', 'Test Error'])
    # graph = plot_frame.plot(x=ind_var_name, y=['Training Error', 'Test Error'], title=model_name + '-' + ind_var_name)
    # graph.set_xlabel(ind_var_name)
    # graph.set_ylabel('Error')
    # plt.ylim(0.0, 0.5)
    # plt.savefig('graphs/' + model_name + '-' + ind_var_name + '.png')


def perform_boosting_analysis(model, name):
    print("Perform analysis for " + name)
    # learning rate
    intervals = 20
    interval_size = 0.5
    section_scores = []
    for i in range(1, intervals + 1):
        learning_rate = i * interval_size
        model.learning_rate = learning_rate
        model.fit(training_features, training_classes)
        predicted_training_classes_section = model.predict(training_features)
        predicted_test_classes_section = model.predict(test_features)
        training_score = accuracy_score(training_classes, predicted_training_classes_section)
        training_error = 1.0 - training_score
        test_score = accuracy_score(test_classes, predicted_test_classes_section)
        test_error = 1.0 - test_score
        section_scores.append([learning_rate, training_error, test_error])

    delta_name = "Learning Rate"
    plot_frame = pd.DataFrame(section_scores, columns=[delta_name, 'Training Error', 'Test Error'])
    graph = plot_frame.plot(x=delta_name, y=['Training Error', 'Test Error'],
                            title=name + '-' + delta_name)
    graph.set_xlabel(delta_name)
    graph.set_ylabel('Error')
    plt.ylim(0.0, 0.5)
    plt.savefig('graphs/' + name + '-' + delta_name + '.png')
    model.learning_rate = 1.0
    # n_estimators
    # intervals = 20
    # interval_size = 25
    # section_scores = []
    # for i in range(1, intervals + 1):
    #     n_estimators = int(i * interval_size)
    #     classifier.n_estimators = n_estimators
    #     classifier.fit(training_features, training_classes)
    #     predicted_training_classes_section = classifier.predict(training_features)
    #     predicted_test_classes_section = classifier.predict(test_features)
    #     training_score = accuracy_score(training_classes, predicted_training_classes_section)
    #     training_error = 1.0 - training_score
    #     test_score = accuracy_score(test_classes, predicted_test_classes_section)
    #     test_error = 1.0 - test_score
    #     section_scores.append([n_estimators, training_error, test_error])
    #
    # plot_frame = pd.DataFrame(section_scores, columns=['Estimators', 'Training Error', 'Test Error'])
    # graph = plot_frame.plot(x='Estimators', y=['Training Error', 'Test Error'],
    #                         title=model_name + '-Estimators')
    # graph.set_xlabel('Estimators')
    # graph.set_ylabel('Error')
    # plt.ylim(0.0, 0.5)
    # plt.savefig('graphs/' + model_name + '-Estimators.png')
    # max_depth
    intervals = 20
    interval_size = 1
    section_scores = []
    for i in range(1, intervals + 1):
        max_depth = int(i * interval_size)
        model.base_estimator.max_depth = max_depth
        model.fit(training_features, training_classes)
        predicted_training_classes_section = model.predict(training_features)
        predicted_test_classes_section = model.predict(test_features)
        training_score = accuracy_score(training_classes, predicted_training_classes_section)
        training_error = 1.0 - training_score
        test_score = accuracy_score(test_classes, predicted_test_classes_section)
        test_error = 1.0 - test_score
        section_scores.append([max_depth, training_error, test_error])

    plot_frame = pd.DataFrame(section_scores, columns=['Max Depth', 'Training Error', 'Test Error'])
    graph = plot_frame.plot(x='Max Depth', y=['Training Error', 'Test Error'],
                            title=name + '-Max Depth')
    graph.set_xlabel('Max Depth')
    graph.set_ylabel('Error')
    plt.ylim(0.0, 0.5)
    plt.savefig('graphs/' + name + '-Max Depth.png')


def perform_svc_poly_analysis(model, name):
    print("Perform analysis for " + name)
    # degree
    intervals = 20
    interval_size = 1
    section_scores = []
    for i in range(1, intervals + 1):
        degree = i * interval_size
        model.degree = degree
        model.fit(training_features, training_classes)
        predicted_training_classes_section = model.predict(training_features)
        predicted_test_classes_section = model.predict(test_features)
        training_score = accuracy_score(training_classes, predicted_training_classes_section)
        training_error = 1.0 - training_score
        test_score = accuracy_score(test_classes, predicted_test_classes_section)
        test_error = 1.0 - test_score
        section_scores.append([degree, training_error, test_error])

    delta_name = "N-Degree"
    plot_frame = pd.DataFrame(section_scores, columns=[delta_name, 'Training Error', 'Test Error'])
    graph = plot_frame.plot(x=delta_name, y=['Training Error', 'Test Error'],
                            title=name + '-' + delta_name)
    graph.set_xlabel(delta_name)
    graph.set_ylabel('Error')
    plt.ylim(0.0, 0.5)
    plt.savefig('graphs/' + name + '-' + delta_name + '.png')


def perform_knearest_neighbors_analysis(model, name):
    print("Perform analysis for " + name)
    # algorithm
    # algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
    # section_scores = []
    # for i in range(0, len(algorithms)):
    #     algorithm = algorithms[i]
    #     classifier.algorithm = algorithm
    #     classifier.fit(training_features, training_classes)
    #     predicted_training_classes_section = classifier.predict(training_features)
    #     predicted_test_classes_section = classifier.predict(test_features)
    #     training_score = accuracy_score(training_classes, predicted_training_classes_section)
    #     training_error = 1.0 - training_score
    #     test_score = accuracy_score(test_classes, predicted_test_classes_section)
    #     test_error = 1.0 - test_score
    #     section_scores.append([algorithm, training_error, test_error])
    #
    # ind_var_name = "Algorithm"
    # plot_frame = pd.DataFrame(section_scores, columns=[ind_var_name, 'Training Error', 'Test Error'])
    # graph = plot_frame.plot(x=ind_var_name, y=['Training Error', 'Test Error'],
    #                         title=model_name + '-' + ind_var_name)
    # graph.set_xlabel(ind_var_name)
    # graph.set_ylabel('Error')
    # plt.ylim(0.0, 0.5)
    # plt.savefig('graphs/' + model_name + '-' + ind_var_name + '.png')
    # p
    # intervals = 3
    # interval_size = 1
    # section_scores = []
    # for i in range(1, intervals + 1):
    #     p = interval_size * i
    #     classifier.p = p
    #     classifier.fit(training_features, training_classes)
    #     predicted_training_classes_section = classifier.predict(training_features)
    #     predicted_test_classes_section = classifier.predict(test_features)
    #     training_score = accuracy_score(training_classes, predicted_training_classes_section)
    #     training_error = 1.0 - training_score
    #     test_score = accuracy_score(test_classes, predicted_test_classes_section)
    #     test_error = 1.0 - test_score
    #     section_scores.append([p, training_error, test_error])
    #
    # ind_var_name = "Power Parameter"
    # plot_frame = pd.DataFrame(section_scores, columns=[ind_var_name, 'Training Error', 'Test Error'])
    # graph = plot_frame.plot(x=ind_var_name, y=['Training Error', 'Test Error'],
    #                         title=model_name + '-' + ind_var_name)
    # graph.set_xlabel(ind_var_name)
    # graph.set_ylabel('Error')
    # plt.ylim(0.0, 0.5)
    # plt.savefig('graphs/' + model_name + '-' + ind_var_name + '.png')


if __name__ == "__main__":
    train = pd.read_csv("data/lendingclub_train.csv", usecols=columns)
    test = pd.read_csv("data/lendingclub_test.csv", usecols=columns)

    training_features = train[columns[:-1]]
    training_classes = train[columns[-1:]].astype(np.bool)
    test_features = test[columns[:-1]]
    test_classes = test[columns[-1:]].astype(np.bool)

    classifiers = [DecisionTreeClassifier(),
                   MLPClassifier(hidden_layer_sizes=10, random_state=7),
                   AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)),
                   SVC(kernel='rbf', gamma='auto'),
                   SVC(kernel='poly', gamma='auto'),
                   KNeighborsClassifier(n_neighbors=1,
                                        weights='distance',
                                        algorithm='brute'),
                   KNeighborsClassifier(n_neighbors=5,
                                        weights='distance',
                                        algorithm='brute'),
                   KNeighborsClassifier(n_neighbors=9,
                                        weights='distance',
                                        algorithm='brute')
                   ]

    model_names = ["DecisionTreeClassifier()",
                   "MLPClassifier",
                   "AdaBoostClassifier(DecisionTreeClassifier())",
                   "SVC(kernel='rbf')",
                   "SVC(kernel='poly')",
                   "KNeighborsClassifier(n_neighbors=1)",
                   "KNeighborsClassifier(n_neighbors=5)",
                   "KNeighborsClassifier(n_neighbors=9)"]

    time_data = []
    error_data = []
    for i in range(0, len(classifiers)):
        classifier = classifiers[i]
        model_name = model_names[i]
        # Time data
        start = timer()
        classifier.fit(training_features, training_classes)
        end = timer()
        training_time = end - start
        start = timer()
        predicted_test_classes = classifier.predict(test_features)
        end = timer()
        classification_time = end - start
        time_data.append([model_name, training_time, classification_time])
        # Error data
        predicted_training_classes = classifier.predict(training_features)
        training_score = accuracy_score(training_classes, predicted_training_classes)
        training_error = 1.0 - training_score
        test_score = accuracy_score(test_classes, predicted_test_classes)
        test_error = 1.0 - test_score
        error_data.append([model_name, training_error, test_error])

        perform_training_size(classifier, model_name)
        perform_iterations(classifier, model_name)

    time_table = pd.DataFrame(time_data,
                              columns=['Model', 'Training Time', 'Classification Time'])
    time_table.set_index('Model', inplace=True)
    time_table.to_csv('tables/Time Data.csv')

    error_table = pd.DataFrame(error_data,
                               columns=['Model', 'Training Error', 'Test Error'])
    error_table.set_index('Model', inplace=True)
    error_table.to_csv('tables/Error Data.csv')

    classifiers = [DecisionTreeClassifier(),
                   MLPClassifier(hidden_layer_sizes=10, random_state=7),
                   AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)),
                   SVC(kernel='rbf', gamma='auto'),
                   SVC(kernel='poly', gamma='auto'),
                   KNeighborsClassifier(n_neighbors=1,
                                        weights='distance',
                                        algorithm='brute'),
                   KNeighborsClassifier(n_neighbors=5,
                                        weights='distance',
                                        algorithm='brute'),
                   KNeighborsClassifier(n_neighbors=9,
                                        weights='distance',
                                        algorithm='brute')
                   ]

    perform_decision_tree_analysis(classifiers[0], model_names[0])
    perform_neural_net_analysis(classifiers[1], model_names[1])
    perform_boosting_analysis(classifiers[2], model_names[2])
    perform_svc_poly_analysis(classifiers[4], model_names[4])
    perform_knearest_neighbors_analysis(classifiers[5], model_names[5])
    perform_knearest_neighbors_analysis(classifiers[6], model_names[6])
    perform_knearest_neighbors_analysis(classifiers[7], model_names[7])
    print("Complete")