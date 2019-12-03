from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import itertools
import operator
import time
from sklearn.datasets import fetch_covtype

"""Ce programme permet de comparer la précision entre les classifieur OvO OvR et Forest"""

# =============================================================================
# MAIN
# =============================================================================

def main():

    newsgroups_train = fetch_covtype()

    # Récupération des data et target
    data = train.csv                                            # newsgroups_train.data
    target = test.csv                                          # newsgroups_train.target
    # Elimination de données pour accélerer le temps de traitement qui ne se terminé pas sur mon PC
    data = data[:len(data)-575000]
    target = target[:len(target)-575000]

    classes = set(target)

    # Créer un jeu de données test et train
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    # Créer une liste contenant des tuples (images,value)
    test_values = [(x_test[index], value) for index, value in enumerate(y_test)]

    start_time = time.time()
    # Créer des classifieur O vs O
    o_vs_o_classifiers = generateOvOClassifier(classes, x_train, y_train)
    print("OvO classifieur Done")
    # Fait les prédictions avec les classifieurs O vs O
    predictOVO(test_values, o_vs_o_classifiers)
    print("Temps d'execution OvO : %s secondes" % (time.time() - start_time))
    print()

    start_time = time.time()
    # Créer des classifieur O vs R
    ovrclassifier = generateOvRClassifier(classes, x_train, y_train)
    print("OvR classifieur Done")
    # Fait les prédictions avec les classifieurs O vs R
    predictOVR(test_values, ovrclassifier)
    print("Temps d'execution OvR : %s secondes" % (time.time() - start_time))
    print()

    start_time = time.time()
    # Créer des forest classifieur
    forestclassifier = RandomForestClassifier(n_estimators=10).fit(x_train, y_train)
    print("Forest classifieur Done")
    # Fait les prédictions avec les forests classifieurs
    predictForest(test_values, forestclassifier)
    print("Temps d'execution Forest : %s secondes" % (time.time() - start_time))
    print()

    start_time = time.time()
    # Créer des SVM classifieur
    SVMclassifier = svm.SVC(gamma='scale', decision_function_shape='ovo', probability=True).fit(x_train, y_train)
    print("SVM classifieur Done")
    # Fait les prédictions avec les SVM classifieurs
    predictSVM(test_values, SVMclassifier)
    print("Temps d'execution SVM : %s secondes" % (time.time() - start_time))
    

# =============================================================================
# FUNCTIONS
# =============================================================================


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Génére des classifeurs pour chaque valeur de notre classe (dans notre cas pour chaque chiffre)
def generateOvRClassifier(classes, x_train, y_train):
    o_vs_r_classifiers = {}
    # Pour chaque chiffre on crée un classifieur O v R
    for elem in classes:
        # On crée une liste contenant toutes les valeurs égale à la valeur que l'on cherche dans ce classifieur
        class_valid = [x_train[index] for index, value in enumerate(y_train) if value == elem]
        # On crée une liste contenant toutes les valeurs différente de la valeur que l'on cherche dans ce classifieur
        class_invalid = [x_train[index] for index, value in enumerate(y_train) if value != elem]
        # On crée un liste avec pour chaque valeur 1 si elle correspond à celle du classifieur, 0 dans le cas contraire
        value = [1] * len(class_valid) + [0] * len(class_invalid)
        # On concatène nos deux liste pour récuperer l'intégralité de nos valeurs
        learn = class_valid + class_invalid
        # On créer un classifieur O v R à partir d'une regression logistique
        o_vs_r_classifiers["%d_rest" % elem] = LogisticRegression(multi_class='ovr',solver='lbfgs').fit(learn, value)
    return o_vs_r_classifiers

# A partir de valeur de test et d'un cassifieur O v R retourne la précision du classifieur
def predictOVR(test_values, o_vs_r_classifiers):
    results = {}
    i=0
    for elem in test_values:
        intern_result = {}
        for name, classifier in o_vs_r_classifiers.items():
            result = classifier.predict([elem[0]])
            result_proba = classifier.predict_proba([elem[0]])
            intern_result[name.split('_')[0]] = result_proba[0][1]
        results[i] = intern_result
        i+=1
    correct = 0
    for key, elem in results.items():
        predicted = max(elem.items(), key=operator.itemgetter(1))[0]
        value = test_values[key][1]
        if int(predicted) == value:
            correct +=1
        #print("Predicted %s and value was %s" %(predicted,value))
    prct = (correct/len(results)*100)
    print(f"Le One versus Rest score a {prct} % de precision")

def generateOvOClassifier(classes, x_train, y_train):
    o_vs_o_classifiers = {}
    # Pour chaque chiffre on crée une combinaison avec chaque autre chiffre pour crée un classifeur O v O
    for elem in itertools.combinations(classes,2):
        # On crée une liste contenant toutes les valeurs égales au premier chiffre de notre combinaison
        class0 = [x_train[index] for index, value in enumerate(y_train) if value == elem[0]]
        # On crée une liste contenant toutes les valeurs égales au second chiffre de notre combinaison
        class1 = [x_train[index] for index, value in enumerate(y_train) if value == elem[1]]
        # On crée un liste avec pour chaque valeur 0 si elle correspond à celle du premier chiffre, 1 si elle correspond au second chiffre
        value = [0] * len(class0) + [1] * len(class1)
        # On concatène nos deux liste pour récuperer l'intégralité des nos valeurs étant égale à l'un des deux chiffre de notre combinaison
        learn = class0 + class1
        # On créer un classifieur O v O à partir d'une regression logistique
        o_vs_o_classifiers['%d_%d'%elem] = LogisticRegression(solver='lbfgs').fit(learn, value)
    return o_vs_o_classifiers

# A partir de valeur de test et d'un cassifieur O v O retourne la précision du classifieur
def predictOVO(test_values, o_vs_o_classifiers):
    """
    TO DO : STATS
    """
    results = {}
    i=0
    for elem in test_values:
        intern_result = {}
        for name,classifiers in o_vs_o_classifiers.items():
            result = classifiers.predict([elem[0]])
            members = name.split('_')
            if intern_result.get(members[result[0]]):
                intern_result[members[result[0]]] += 1
            else:
                intern_result[members[result[0]]] = 1
        results[i] = intern_result
        i+=1
    correct = 0
    for key,elem in results.items():
        predicted = max(elem.items(), key=operator.itemgetter(1))[0]
        value = test_values[key][1]
        if int(predicted) == value:
            correct += 1
        #print("Predicted %s and value was %s" %(predicted,value))
    prct = (correct/len(results)*100)
    print(f"Le One versus One score a {prct} % de precision")

# A partir de valeur de test et d'un forest cassifieur retourne la précision du classifieur
def predictForest(test_values, forest_classifiers):
    correct = 0
    for elem in test_values:
        # utilisation du classifieur
        result = forest_classifiers.predict([elem[0]])
        # Compare si la prédiction correspond au résultats réel
        if (elem[1]==result):
            correct +=1
    prct = (correct/len(test_values)*100)
    print(f"Le Forest score a {prct} % de precision")

# A partir de valeur de test et d'un forest classifieur retourne la précision du classifieur
def predictSVM(test_values, SVM_classifiers):
    correct = 0
    for elem in test_values:
        # utilisation du classifieur
        result = SVM_classifiers.predict([elem[0]])
        # Compare si la prédiction correspond au résultats réel
        if (elem[1]==result):
            correct +=1
    prct = (correct/len(test_values)*100)
    print(f"Le SVM score a {prct} % de precision")
        


# =============================================================================
# SCRIPT INITIATE
# =============================================================================

if __name__ == '__main__':
    main()
