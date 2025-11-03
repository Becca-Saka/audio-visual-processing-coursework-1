import matplotlib.pyplot as plt
import pandas as pd



def plot_bar_chart(output_name, results):
    df = pd.DataFrame(results)

    df['Correct'] = df['Actual Name'].str.lower() == df['Predicted Name'].str.lower()

    accuracy_counts = df.groupby('Actual Name')['Correct'].sum().reset_index()
    accuracy_counts.rename(columns={'Correct': 'Correct Predictions'}, inplace=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(accuracy_counts['Actual Name'], accuracy_counts['Correct Predictions'], color='skyblue')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, int(yval), ha='center', va='bottom', fontsize=10)

    plt.title('Correct Predictions per Person')
    plt.xlabel('Person')
    plt.ylabel('Number of Correct Predictions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
