Selected features: ['acc_y', 'pca_2', 'acc_x_roll', 'acc_y_roll', 'acc_z_roll', 'gyro_z_roll']
Accuracy score: 0.9640618470539072

Selected features: ['acc_x', 'gyro_x', 'gyro_z', 'lda_1', 'lda_2', 'acc_x_roll', 'acc_y_roll', 'acc_z_roll', 'gyro_x_roll', 'gyro_z_roll']
Accuracy score: 0.9619724195570414

Feature  Importance
      pca_1    0.238153
 acc_y_roll    0.207583
      lda_3    0.179591
 acc_x_roll    0.161151
gyro_z_roll    0.049055
 acc_z_roll    0.032695
gyro_x_roll    0.021914
     gyro_z    0.019712
      acc_y    0.016852
      lda_1    0.013294
      acc_x    0.010143
      lda_2    0.009517
     gyro_y    0.009134
     gyro_x    0.009012
      pca_2    0.008695
gyro_y_roll    0.008554
      pca_3    0.002912
      acc_z    0.002033


Random forest without participant A
{'clf__bootstrap': False, 'clf__max_depth': 30, 'clf__min_samples_leaf': 1, 'clf__min_samples_split': 2, 'clf__n_estimators': 50}
Accuracy on participant A: 0.9783211494832367
Classification Report:
               precision    recall  f1-score   support

       bench       0.82      1.00      0.90       370
        dead       1.00      0.99      1.00       709
         ohp       1.00      0.92      0.96       963
        rest       1.00      1.00      1.00       984
         row       0.97      1.00      0.99       113
       squat       1.00      1.00      1.00       828

    accuracy                           0.98      3967
   macro avg       0.97      0.98      0.97      3967
weighted avg       0.98      0.98      0.98      3967

RandomForest	feature set6	{'clf__bootstrap': False, 'clf__max_depth': None, 'clf__min_samples_leaf': 1, 'clf__min_samples_split': 2, 'clf__n_estimators': 50}