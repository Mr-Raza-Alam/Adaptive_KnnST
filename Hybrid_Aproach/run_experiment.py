import subprocess
import sys

def main():
    print("========================================")
    print("  Missing Data Imputation Experiments   ")
    print("========================================")
    print("Select Dataset:")
    print("[1] Seoul Air Quality (PM2.5, PM10, NO2)")
    print("[2] Crowd Temperature")
    print("========================================")
    dataset_choice = input("Enter the number of the dataset to use: ").strip()
    
    if dataset_choice == "1":
        dataset_arg = "seoul"
    elif dataset_choice == "2":
        dataset_arg = "temperature"
    else:
        print("\nInvalid dataset choice. Exiting.\n")
        return

    print("\n========================================")
    print("Select Comparison:")
    print("[1] Base KNN-ST vs. Hybrid KNN-ST")
    print("[2] Compressive Sensing (CS) vs. Hybrid KNN-ST")
    print("[3] Regression Baselines (Marchang 2022) vs. Hybrid KNN-ST")
    print("[4] Regression Baselines (Marchang 2022) vs. Base KNN-ST")
    print("[5] GNN vs. Hybrid KNN-ST (Coming Soon)")
    print("========================================")
    
    choice = input("Enter the number of the experiment to run: ").strip()
    
    if choice == "1":
        print(f"\nStarting Base KNN-ST vs. Hybrid KNN-ST experiment on {dataset_arg} dataset...\n")
        subprocess.run([sys.executable, "knn_st_vs_hybrid.py", "--dataset", dataset_arg])
    elif choice == "2":
        print(f"\nStarting Compressive Sensing (CS) vs. Hybrid KNN-ST experiment on {dataset_arg} dataset...\n")
        subprocess.run([sys.executable, "cs_vs_hybrid.py", "--dataset", dataset_arg])
    elif choice == "3":
        print(f"\nStarting Regression Baselines vs. Hybrid KNN-ST experiment on {dataset_arg} dataset...\n")
        subprocess.run([sys.executable, "regression_vs_hybrid.py", "--dataset", dataset_arg])
    elif choice == "4":
        print(f"\nStarting Regression Baselines vs. Base KNN-ST experiment on {dataset_arg} dataset...\n")
        subprocess.run([sys.executable, "regression_vs_base.py", "--dataset", dataset_arg])
    elif choice == "5":
        print("\nGNN comparison is not yet implemented. Please check back later!\n")
    else:
        print("\nInvalid choice. Exiting.\n")

if __name__ == "__main__":
    main()
