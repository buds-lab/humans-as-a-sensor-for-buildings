
def normalise_total_cozie(dataframe, group, threshold):
    cluster_df = dataframe.copy(deep=True)
    # remapping
    cluster_df['prefer_cooler'] = cluster_df['thermal_cozie'][cluster_df.thermal_cozie == 11.0] #Too Hot
    cluster_df['prefer_warmer'] = cluster_df['thermal_cozie'][cluster_df.thermal_cozie == 9.0] # Too COld
    cluster_df['thermaly_comfy'] = cluster_df['thermal_cozie'][cluster_df.thermal_cozie == 10.0] #_cOmfy

    
    cluster_df['prefer_dimmer'] = cluster_df['light_cozie'][cluster_df.light_cozie == 11.0] # Too bright
    cluster_df['prefer_brighter'] = cluster_df['light_cozie'][cluster_df.light_cozie == 9.0] # Too Dark
    cluster_df['visually_comfy'] = cluster_df['light_cozie'][cluster_df.light_cozie == 10.0] #_comfy
    
    cluster_df['prefer_quieter'] = cluster_df['noise_cozie'][cluster_df.noise_cozie == 11.0] # Too Loud
    cluster_df['prefer_louder'] = cluster_df['noise_cozie'][cluster_df.noise_cozie == 9.0] # Too Quiet
    cluster_df['aurally_comfy'] = cluster_df['noise_cozie'][cluster_df.noise_cozie == 10.0] #_comfy

    

    # group
    group_df=cluster_df.groupby(group).count()
    group_df = group_df[group_df.thermal_cozie >= threshold]

    group_df[['prefer_cooler', 'prefer_warmer', 'thermaly_comfy']] = group_df[['prefer_cooler', 'prefer_warmer', 'thermaly_comfy']].div(group_df.thermal_cozie, axis=0)
    group_df[['prefer_dimmer', 'prefer_brighter', 'visually_comfy']] = group_df[['prefer_dimmer', 'prefer_brighter','visually_comfy']].div(group_df.light_cozie, axis=0) 
    group_df[['prefer_quieter', 'prefer_louder', 'aurally_comfy']] = group_df[['prefer_quieter', 'prefer_louder', 'aurally_comfy']].div(group_df.noise_cozie, axis=0)



    return (group_df)

def find_optimal_tree_depth(clf, train_vectors, train_labels, plot=True):
    """Choose the optimal depth of a tree model 
    """
    
    DEFAULT_K = 10
    
    # generate a list of potential depths to calculate the optimal
    depths = list(range(1, 25))

    # empty list that will hold cv scores
    cv_scores = []

    print("Finding optimal tree depth")
    # find optimal tree depth    
    for d in depths: # TODO: try using chooseK(train_labels) instead of jus DEFAULT_K
        clf_depth = clf.set_params(max_depth = d) # use previous parameters while changing depth

        scores = cross_val_score(clf_depth, train_vectors, 
                                 train_labels, cv = choose_k(train_labels),
                                 scoring = 'accuracy') # accuracy here is f1 micro
        cv_scores.append(scores.mean())

    # changing to misclassification error and determining best depth
    MSE = [1 - x for x in cv_scores] # MSE = 1 - f1_micro
    optimal_depth = depths[MSE.index(min(MSE))]
    
    print("The optimal depth is: {}".format(optimal_depth))
    print("Expected accuracy (f1 micro) based on Cross-Validation: {}".format(cv_scores[depths.index(optimal_depth)]))
    
    if plot:
        # plot misclassification error vs depths
        fig = plt.figure(figsize=(12, 10))
        plt.plot(depths, MSE)
        plt.xlabel('Tree Depth', fontsize = 20)
        plt.ylabel('Misclassification Error', fontsize = 20)
#         plt.legend(fontsize = 15)
        plt.show()

    return optimal_depth