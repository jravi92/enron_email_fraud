#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def fill_NAN_values():
    # Update NaN values with 0 except for email 
    people_keys = data_dict.keys()
    feature_keys = data_dict[people_keys[0]]
    NAN_features = {}
    # Get NaN values and replace it
    for feature in feature_keys:
        NAN_features[feature] = 0
    for person in people_keys:
        for feature in feature_keys:
            if feature != 'email_address' and \
                data_dict[person][feature] == 'NaN':
                data_dict[person][feature] = 0
                NAN_features[feature] += 1

    return NAN_features

def poi_missing_email_info():
    poi_count = 0
    poi_keys = []
    for person in data_dict.keys():
        if data_dict[person]["poi"]:
            poi_count += 1
            poi_keys.append(person)

    poi_missing_emails = []
    for poi in poi_keys:
        if (data_dict[poi]['to_messages'] == 'NaN' and data_dict[poi]['from_messages'] == 'NaN') or \
            (data_dict[poi]['to_messages'] == 0 and data_dict[poi]['from_messages'] == 0):
            poi_missing_emails.append(poi)

    return poi_count, poi_missing_emails

def scatter_plot(x_feature, y_feature):
    features = ['poi', x_feature, y_feature]
    data = featureFormat(data_dict, features)

    
    for point in data:
        x = point[1]
        y = point[2]
        if point[0]:
            plt.scatter(x, y, color="b", marker="o")
        else:
            plt.scatter(x, y, color='r', marker="x")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    pic = x_feature + y_feature + '.png'
    plt.savefig(pic, transparent=True)
    #plt.show()

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'deferral_payments',
                 'total_payments',
                 'loan_advances',
                 'bonus',
                 'restricted_stock_deferred',
                 'deferred_income',
                 'total_stock_value',
                 'expenses',
                 'exercised_stock_options',
                 'long_term_incentive',
                 'restricted_stock',
                 'director_fees',
                 'to_messages',
                 'from_poi_to_this_person',
                 'from_messages',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
print '============ Load Dataset ==============='
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print '========= Dataset Overview =============='
people_keys = data_dict.keys()
feature_keys = data_dict[people_keys[0]]
poi_c, poi_missing_emails = poi_missing_email_info()

print 'Number of people in dataset: %d' % len(people_keys)
print 'Number of POIs in dataset: %d out of 34 total POIs' % poi_c
print 'Number of non-POIs in dataset: %d' % (len(people_keys) - poi_c)
print 'POIs with zero or missing to/from email messages in dataset: %d' % len(poi_missing_emails)
print poi_missing_emails

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
print '\n'
print '========== Removing Outliers ============'
features_with_NAN = fill_NAN_values()
print 'Updating NaN values in features'
print features_with_NAN

# Outlier at 26M in salary -> 'total
scatter_plot('salary', 'bonus')
# Remove outlier 'total'
print 'Scatter plot shows large outlier in salary'
print 'Outlier is "Total" value and should be removed'
data_dict.pop('TOTAL')

# Investigate other high salary or bonuses for outliers
fortuner = []
people_keys = data_dict.keys()
for person in people_keys:
    if data_dict[person]["bonus"] > 5000000 or data_dict[person]["salary"] > 2000000:
        fortuner.append(person)


high_salary_bonus = fortuner

print 'Salary Bonus Fortuner (2M+ and 5M+): \n', high_salary_bonus

# Only 146 values, can visually review names
print 'Look for other "odd" values to remove'
print people_keys
print 'Found name: "THE TRAVEL AGENCY IN THE PARK" '

data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

scatter_plot('salary', 'bonus')



# Create new features
print '\n'
print '============ Create Features ==================='

people_keys = data_dict.keys()

for person in people_keys:
    to_poi = float(data_dict[person]['from_this_person_to_poi'])
    from_poi = float(data_dict[person]['from_poi_to_this_person'])
    to_msg_total = float(data_dict[person]['to_messages'])
    from_msg_total = float(data_dict[person]['from_messages'])

    #print to_poi,from_poi, to_msg_total, from_msg_total
    #print "\n"
    if from_msg_total > 0:
        data_dict[person]['to_poi_fraction'] = to_poi / from_msg_total
    else:
        data_dict[person]['to_poi_fraction'] = 0

    if to_msg_total > 0:
        data_dict[person]['from_poi_fraction'] = from_poi / to_msg_total
    else:
        data_dict[person]['from_poi_fraction'] = 0

# fraction of your salary represented by your bonus (or something like that)
    person_salary = float(data_dict[person]['salary'])
    person_bonus = float(data_dict[person]['bonus'])
    if person_salary > 0 and person_bonus > 0:
        data_dict[person]['salary_bonus_fraction'] = data_dict[person]['salary'] / data_dict[person]['bonus']
    else:
        data_dict[person]['salary_bonus_fraction'] = 0

# Add new feature to features_list
features_list.extend(['to_poi_fraction', 'from_poi_fraction', 'salary_bonus_fraction'])

print 'Updated features_list: \n', features_list


my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



# Feature selection
print '\n'
print '============= Feature Selection ==================='
# Feature select is performed with SelectKBest where k is selected by GridSearchCV
# Using Stratify for small and minority POI dataset


from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
# Extract features and labels from dataset for local testing
from sklearn.feature_selection import SelectKBest

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, train_size=.45, stratify=labels)




##################################
from sklearn.model_selection import GridSearchCV

n_features = np.arange(1, len(features_list))

# Create a pipeline with feature selection and classification
pipe = Pipeline([
    ('select_features', SelectKBest()),
    ('classify', AdaBoostClassifier())
])

param_grid = [
    {
        'select_features__k': n_features
    }
]

# Use GridSearchCV to automate the process of finding the optimal number of features
ada_clf= GridSearchCV(pipe, param_grid=param_grid, scoring='accuracy', cv = 10)
ada_clf.fit(features, labels)
print 'Best parameters: ', ada_clf.best_params_
##################################






skbest = SelectKBest(k=11 )  # try best value to fit
sk_transform = skbest.fit_transform(features_train, labels_train)
indices = skbest.get_support(True)
print indices
print skbest.scores_

n_list = ['poi']
for index in indices:
    print 'features: %s score: %f' % (features_list[index + 1], skbest.scores_[index])
    n_list.append(features_list[index + 1])



n_list = ['poi', 'salary', 'total_stock_value', 'expenses', 'bonus',
          'exercised_stock_options', 'deferred_income',
          'to_poi_fraction', 'from_poi_to_this_person', 'from_poi_fraction',
          'long_term_incentive','total_payments','shared_receipt_with_poi' ]



features_list = n_list
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf_gau = GaussianNB()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from tester import dump_classifier_and_data, test_classifier
from sklearn.linear_model import LogisticRegression

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, class_weight='balanced'),
                         n_estimators=50, learning_rate=.8)

clf2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, class_weight='balanced'),
                        n_estimators=40, learning_rate=.6)

clf3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, class_weight='balanced'),
                        n_estimators=30, learning_rate=.4)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=12, min_samples_split=6)
clf_log = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
# Example starting point. Try investigating other evaluation techniques!
##from sklearn.cross_validation import train_test_split
##features_train, features_test, labels_train, labels_test = \
##train_test_split(features, labels, test_size=0.3, random_state=42)

test_classifier(clf, my_dataset, features_list)
test_classifier(clf2, my_dataset, features_list)
test_classifier(clf3, my_dataset, features_list)
test_classifier(clf_rf, my_dataset, features_list)
test_classifier(clf_gau, my_dataset, features_list)
test_classifier(clf_log, my_dataset, features_list)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
