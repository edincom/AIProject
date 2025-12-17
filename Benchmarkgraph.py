# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set style
# sns.set_style("whitegrid")
# plt.rcParams['figure.figsize'] = (14, 10)

# # Load the CSV
# df = pd.read_csv('benchmark_results_summary.csv')

# # Calculate overall averages by configuration
# overall_avg = df.groupby('configuration')[['accuracy', 'completeness', 
#                                             'citation_quality', 'relevance', 
#                                             'overall']].mean()

# # Create figure with subplots
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# fig.suptitle('RAG Configuration Benchmark Results', fontsize=16, fontweight='bold')

# # 1. Overall Performance Bar Chart
# ax1 = axes[0, 0]
# overall_avg['overall'].sort_values(ascending=False).plot(kind='bar', ax=ax1, color='skyblue')
# ax1.set_title('Overall Performance by Configuration', fontweight='bold')
# ax1.set_xlabel('Configuration')
# ax1.set_ylabel('Overall Score (1-5)')
# ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
# ax1.axhline(y=overall_avg['overall'].mean(), color='red', linestyle='--', label='Average')
# ax1.legend()

# # 2. Criteria Comparison Heatmap
# ax2 = axes[0, 1]
# criteria_data = overall_avg[['accuracy', 'completeness', 'citation_quality', 'relevance']]
# sns.heatmap(criteria_data.T, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax2, cbar_kws={'label': 'Score'})
# ax2.set_title('Criteria Performance Heatmap', fontweight='bold')
# ax2.set_xlabel('Configuration')
# ax2.set_ylabel('Criteria')

# # 3. Performance by Topic (Overall Score)
# ax3 = axes[1, 0]
# topic_pivot = df.pivot_table(values='overall', index='topic', columns='configuration')
# topic_pivot.plot(kind='bar', ax=ax3)
# ax3.set_title('Overall Performance by Topic', fontweight='bold')
# ax3.set_xlabel('Topic')
# ax3.set_ylabel('Overall Score (1-5)')
# ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
# ax3.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# # 4. Chunk Size Comparison
# ax4 = axes[1, 1]
# # Extract chunk sizes and types
# df['chunk_size'] = df['configuration'].str.extract(r'(\d+)').astype(int)
# df['rag_type'] = df['configuration'].str.extract(r'(Basic|Advanced)')

# chunk_comparison = df.groupby(['chunk_size', 'rag_type'])['overall'].mean().reset_index()
# for rag_type in chunk_comparison['rag_type'].unique():
#     data = chunk_comparison[chunk_comparison['rag_type'] == rag_type]
#     ax4.plot(data['chunk_size'], data['overall'], marker='o', label=rag_type, linewidth=2)

# ax4.set_title('Chunk Size Impact on Performance', fontweight='bold')
# ax4.set_xlabel('Chunk Size')
# ax4.set_ylabel('Overall Score (1-5)')
# ax4.legend()
# ax4.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
# print("✓ Graph saved to 'benchmark_results.png'")
# plt.show()


#________________________________________________

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv('benchmark_results_summary.csv')

# Calculate overall averages by configuration
overall_avg = df.groupby('configuration')['overall'].mean()

# Create figure
plt.figure(figsize=(10, 6))

# Bar chart
overall_avg.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Overall Performance by Configuration', fontsize=16, fontweight='bold')
plt.xlabel('Configuration', fontsize=12)
plt.ylabel('Overall Score (1-5)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.axhline(y=overall_avg.mean(), color='red', linestyle='--', linewidth=2, label=f'Average ({overall_avg.mean():.2f})')
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig('overall_performance.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved to 'overall_performance.png'")
plt.show()