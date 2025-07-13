from functions.LOOCV import *

class PlotEvaluation(LOOCV_performance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def check_ttest(self, individual_distribution, conventional_distribution, conventional_distribution_dh=None, individual_distribution_dh=None):
        """
        Check the statistical significance of the differences between the classification performance of the different GMAC thresholds applied using paired t-test.
        Note:
        - The null hypothesis for the Shapiro-Wilk test is that the data is normally distributed.
        - The null hypothesis for the paired t-tests is that there is no significant difference between the paired samples.
        """
        # Check for normality using Shapiro-Wilk test
        _, p_conventional = shapiro(conventional_distribution)
        _, p_conventional_dh = shapiro(conventional_distribution_dh)
        _, p_individual = shapiro(individual_distribution)
        _, p_individual_dh = shapiro(individual_distribution_dh)
        
        if p_individual > 0.05 and p_conventional > 0.05 and p_conventional_dh > 0.05 and p_individual_dh > 0.05:
            print("The data is normally distributed.")
        else:
            print("The data is not normally distributed.")
        
        # Perform paired t-test
        _, p_value_individual_conventional = ttest_rel(individual_distribution, conventional_distribution)
        _, p_value_individual_conventional_dh = ttest_rel(individual_distribution_dh, conventional_distribution_dh)
        print(f"Paired t-test individual vs conventional: {p_value_individual_conventional}")
        print(f"Paired t-test individual vs conventional DH: {p_value_individual_conventional_dh}")

        return p_value_individual_conventional, p_value_individual_conventional_dh

    def print_classification_performance(self, individual, conventional, conventional_dh, individual_dh, metric='ROCAUC'):
        """
        Print the classification performance metrics for the different GMAC thresholds.
        Parameters:
        """
        print(f"Individual GMAC threshold affected {metric}: {np.mean(individual):.2f}, standard deviation: {np.std(individual):.2f}")
        print(f"Conventional GMAC threshold affected {metric}: {np.mean(conventional):.2f}, standard deviation: {np.std(conventional):.2f}")
        print(f"Individual GMAC threshold unaffected {metric}: {np.mean(individual_dh):.2f}, standard deviation: {np.std(individual_dh):.2f}")
        print(f"Conventional GMAC threshold unaffected {metric}: {np.mean(conventional_dh):.2f}, standard deviation: {np.std(conventional_dh):.2f}")

    def plot_significance_stars(self, ax, bracket_positions, p_values, bracket_heights, position="above"):
        """
        Adds significance stars to the plot, either above, below, or alternating above/below the boxplots.
        *(p < 0.05), **(p < 0.01), ***(p < 0.001)
        position: "above", "below", or "below_above" (alternating for each bracket)
        """
        for idx, ((start, end), p_val, y) in enumerate(zip(bracket_positions, p_values, bracket_heights)):
            x1, x2 = start, end  # x-coordinates of the brackets
            h, col = 0.02, 'k'   # Adjust height and color of the bracket

            def draw_star(pos, y_val):
                if p_val < 0.001:
                    ax.text((x1 + x2) * .5, y_val + 1.1*h if pos == "above" else y_val - 2.5*h, '***', ha='center', va='bottom', color=col, fontsize=12)
                elif p_val < 0.01:
                    ax.text((x1 + x2) * .5, y_val + 1.1*h if pos == "above" else y_val - 2.5*h, '**', ha='center', va='bottom', color=col, fontsize=12)
                elif p_val < 0.05:
                    ax.text((x1 + x2) * .5, y_val + 1.1*h if pos == "above" else y_val - 2.5*h, '*', ha='center', va='bottom', color=col, fontsize=12)

            def draw_bracket(pos, y_val):
                if pos == "above":
                    ax.plot([x1, x1, x2, x2], [y_val, y_val+h, y_val+h, y_val], lw=1.0, c=col)
                elif pos == "below":
                    ax.plot([x1, x1, x2, x2], [y_val, y_val-h, y_val-h, y_val], lw=1.0, c=col)

            if position == "above":
                draw_star("above", y)
                draw_bracket("above", y)
            elif position == "below":
                draw_star("below", y)
                draw_bracket("below", y)
            elif position == "below_above":
                if idx % 2 == 0:
                    draw_star("below", y)
                    draw_bracket("below", y)
                else:
                    draw_star("above", y)
                    draw_bracket("above", y)           

    def plot_AUC(self, significnce_brackets='pvalues'):
        colors = thesis_style.get_thesis_colours()
        ttest_pvalue_individual_conventional, ttest_pvalue_individual_conventional_dh = self.check_ttest(
            self.individual_AUC_list_ndh, self.conventional_AUC_list_ndh, self.conventional_AUC_list_dh, self.individual_AUC_list_dh
        )

        mean_markers = dict(marker='D', markerfacecolor=colors['black'], markersize=6, markeredgewidth=0)
        meadian_markers = dict(color=colors['black_grey'])

        self.print_classification_performance(self.individual_AUC_list_ndh, self.conventional_AUC_list_ndh, self.conventional_AUC_list_dh, self.individual_AUC_list_dh, metric='ROCAUC')

        fig, ax = plt.subplots(figsize=(10, 6))

        # AUC is clinically useful (≥0.75) according to [Fan et al., 2006]
        ax.axhline(y=0.75, color=colors['grey'], linestyle='dotted', label='Clinically required performance [Fan et al., 2006]', lw=2.0)
        # random classifier
        ax.axhline(y=0.5, color=colors['black_grey'], linestyle='--', label='Random classifier', lw=1.3)
        ax.add_artist(plt.legend(loc='upper left', frameon=False, fontsize=10))

        box_conventional = ax.boxplot(self.conventional_AUC_list_ndh, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.4)
        box_individual = ax.boxplot(self.individual_AUC_list_ndh, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.4)
        box_conventional_dh = ax.boxplot(self.conventional_AUC_list_dh, positions=[3], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.4)
        box_individual_dh = ax.boxplot(self.individual_AUC_list_dh, positions=[4], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.4)

        # Set box colors
        for box in box_conventional['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_individual['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_conventional_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)
        for box in box_individual_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)

        # Add colors of healthy and affected boxes to legend
        legend_colors = [colors['healthy'], colors['affected']]
        legend_labels = ['Unaffected side', 'Affected side']
        legend_patches = [mpatches.Patch(color=color, label=label, alpha=0.7) for color, label in zip(legend_colors, legend_labels)]
        ax.add_artist(plt.legend(handles=legend_patches, frameon=False, loc='upper right', reverse=True))

        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['Conventional\nthresholds', 'Individual\nthresholds', 'Conventional\nthresholds', 'Individual\nthresholds'], fontsize=10)
        ax.set_ylim(0.45, 1.0)

        plt.rcParams.update({'font.size': 12})
        plt.ylabel('ROC AUC')
        plt.title('Functional movement detection performance')

        # Define significance bracket positions, p-values, and heights
        bracket_heights = [0.59, 0.85]  # Different heights for the brackets above the boxplot
        bracket_positions = [(1, 2), (3, 4)]  # (start, end) of the brackets
        p_values = [ttest_pvalue_individual_conventional, ttest_pvalue_individual_conventional_dh]
        if significnce_brackets == 'stars':
            self.plot_significance_stars(ax, bracket_positions, p_values, bracket_heights, position="below_above")
        else:
            self.plot_significance_brackets(ax, bracket_positions, p_values, bracket_heights, position="above")

        plt.savefig(os.path.join(save_path.downloadsPath, 'ROC_AUC.pdf'), bbox_inches='tight')
        plt.show()

    def plot_Accuracy(self, significnce_brackets='pvalues'):
        colors = thesis_style.get_thesis_colours()
        ttest_pvalue_individual_conventional, ttest_pvalue_individual_conventional_dh = self.check_ttest(
            self.individual_accuracy_list_ndh, self.conventioanl_accuracy_list_ndh, self.conventional_accuracy_list_dh, self.individual_accuracy_list_dh
        )

        mean_markers = dict(marker='D', markerfacecolor=colors['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=colors['black_grey'])

        self.print_classification_performance(self.individual_accuracy_list_ndh, self.conventioanl_accuracy_list_ndh, self.conventional_accuracy_list_dh, self.individual_accuracy_list_dh, metric='Accuracy')

        fig, ax = plt.subplots(figsize=(12, 6))

        box_conventional = ax.boxplot(self.conventioanl_accuracy_list_ndh, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_individual = ax.boxplot(self.individual_accuracy_list_ndh, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_conventional_dh = ax.boxplot(self.conventional_accuracy_list_dh, positions=[3], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_individual_dh = ax.boxplot(self.individual_accuracy_list_dh, positions=[4], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)

        # Set box colors
        for box in box_conventional['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_individual['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_conventional_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)
        for box in box_individual_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)

        # Accuracy is clinically useful (≥90%) according to [Lang et al., 2020]
        ax.axhline(y=0.9, color=colors['grey'], linestyle='dotted', label='Clinically required performance [Lang et al., 2020]', lw=3.0)
        # random classifier
        ax.axhline(y=0.5, color=colors['black_grey'], linestyle='--', label='Performance of random classifier', lw=2.0)
        ax.add_artist(plt.legend(loc='upper right'))

        # Add colors of healthy and affected boxes to legend
        legend_colors = [colors['healthy'], colors['affected']]
        legend_labels = ['Unaffected side', 'Affected side']
        legend_patches = [mpatches.Patch(color=color, label=label, alpha=0.7) for color, label in zip(legend_colors, legend_labels)]
        ax.add_artist(plt.legend(handles=legend_patches, loc='lower right', reverse=True))

        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['Conventional\nthresholds', 'Individual\nthresholds', 'Conventional\nthresholds', 'Individual\nthresholds'], fontsize=10)
        ax.set_ylim(0.45, 1.0)

        plt.rcParams.update({'font.size': 12})
        plt.ylabel('Accuracy')
        plt.title('Functional movement detection performance')

        # Define significance bracket positions, p-values, and heights
        bracket_heights = [0.85, 0.85]  # Different heights for the brackets above the boxplot
        bracket_positions = [(1, 2), (3, 4)]  # (start, end) of the brackets
        p_values = [ttest_pvalue_individual_conventional, ttest_pvalue_individual_conventional_dh]
        if significnce_brackets == 'stars':
            bracket_heights = [0.85, 0.85]
            bracket_positions = [(1, 2), (3, 4)]
            self.plot_significance_stars(ax, bracket_positions, p_values, bracket_heights, position="below")
        else:
            self.plot_significance_brackets(ax, bracket_positions, p_values, bracket_heights, position="above")

        plt.savefig(os.path.join(save_path.downloadsPath, 'Accuracy.pdf'), bbox_inches='tight')
        plt.show()

    def plot_F1(self, significnce_brackets='pvalues'):
        colors = thesis_style.get_thesis_colours()
        ttest_pvalue_individual_conventional, ttest_pvalue_individual_conventional_dh = self.check_ttest(
            self.individual_F1_list_ndh, self.conventioanl_F1_list_ndh, self.conventional_F1_list_dh, self.individual_F1_list_dh
        )

        mean_markers = dict(marker='D', markerfacecolor=colors['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=colors['black_grey'])

        self.print_classification_performance(self.individual_F1_list_ndh, self.conventioanl_F1_list_ndh, self.conventional_F1_list_dh, self.individual_F1_list_dh, metric='F1')

        fig, ax = plt.subplots(figsize=(12, 6))

        box_conventional = ax.boxplot(self.conventioanl_F1_list_ndh, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_individual = ax.boxplot(self.individual_F1_list_ndh, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_conventional_dh = ax.boxplot(self.conventional_F1_list_dh, positions=[3], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_individual_dh = ax.boxplot(self.individual_F1_list_dh, positions=[4], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)

        # Set box colors
        for box in box_conventional['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_individual['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_conventional_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)
        for box in box_individual_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)
        
        # Add colors of healthy and affected boxes to legend
        legend_colors = [colors['healthy'], colors['affected']]
        legend_labels = ['Unaffected side', 'Affected side']
        legend_patches = [mpatches.Patch(color=color, label=label, alpha=0.7) for color, label in zip(legend_colors, legend_labels)]
        ax.add_artist(plt.legend(handles=legend_patches, loc='lower right', reverse=True))

        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['Conventional\nthresholds', 'Individual\nthresholds', 'Conventional\nthresholds', 'Individual\nthresholds'], fontsize=10)
        ax.set_ylim(-0.05, 1.0)

        plt.rcParams.update({'font.size': 12})
        plt.ylabel('F1 Score')
        plt.title('Functional movement detection performance')

        # Define significance bracket positions, p-values, and heights
        bracket_heights = [0.85, 0.92]  # Different heights for the brackets above the boxplot
        bracket_positions = [(1, 2), (3, 4)]  # (start, end) of the brackets
        p_values = [ttest_pvalue_individual_conventional, ttest_pvalue_individual_conventional_dh]
        if significnce_brackets == 'stars':
            bracket_heights = [0.85, 0.92]
            bracket_positions = [(1, 2), (3, 4)]
            self.plot_significance_stars(ax, bracket_positions, p_values, bracket_heights, position="below")
        else:
            self.plot_significance_brackets(ax, bracket_positions, p_values, bracket_heights, position="above")
        
        plt.savefig(os.path.join(save_path.downloadsPath, 'F1_Score.pdf'), bbox_inches='tight')
        plt.show()
    
    def plot_YI(self, significnce_brackets='pvalues'):
        colors = thesis_style.get_thesis_colours()
        ttest_pvalue_individual_conventional, ttest_pvalue_individual_conventional_dh = self.check_ttest(
            self.individual_YI_list_ndh, self.conventioanl_YI_list_ndh, self.conventional_YI_list_dh, self.individual_YI_list_dh
        )

        mean_markers = dict(marker='D', markerfacecolor=colors['black'], markersize=5, markeredgewidth=0)
        meadian_markers = dict(color=colors['black_grey'])

        self.print_classification_performance(self.individual_YI_list_ndh, self.conventioanl_YI_list_ndh, self.conventional_YI_list_dh, self.individual_YI_list_dh, metric='Youden Index')

        fig, ax = plt.subplots(figsize=(12, 6))

        box_conventional = ax.boxplot(self.conventioanl_YI_list_ndh, positions=[1], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_individual = ax.boxplot(self.individual_YI_list_ndh, positions=[2], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_conventional_dh = ax.boxplot(self.conventional_YI_list_dh, positions=[3], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)
        box_individual_dh = ax.boxplot(self.individual_YI_list_dh, positions=[4], showmeans=True, patch_artist=True, meanprops=mean_markers, medianprops=meadian_markers, widths=0.3)

        # Set box colors
        for box in box_conventional['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_individual['boxes']:
            box.set(facecolor=colors['affected'], alpha=0.7)
        for box in box_conventional_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)
        for box in box_individual_dh['boxes']:
            box.set(facecolor=colors['healthy'], alpha=0.7)

        # random classifier
        ax.axhline(y=0.0, color=colors['black_grey'], linestyle='--', label='Performance of random classifier', lw=2.0)
        ax.add_artist(plt.legend(loc='upper right'))

        # Add colors of healthy and affected boxes to legend
        legend_colors = [colors['healthy'], colors['affected']]
        legend_labels = ['Unaffected side', 'Affected side']
        legend_patches = [mpatches.Patch(color=color, label=label, alpha=0.7) for color, label in zip(legend_colors, legend_labels)]
        ax.add_artist(plt.legend(handles=legend_patches, loc='lower right', reverse=True))

        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['Conventional\nthresholds', 'Individual\nthresholds', 'Conventional\nthresholds', 'Individual\nthresholds'], fontsize=10)
        ax.set_ylim(-0.05, 1.0)

        plt.rcParams.update({'font.size': 12})
        plt.ylabel('Youden Index')
        plt.title('Functional movement detection performance')

        # Define significance bracket positions, p-values, and heights
        bracket_heights = [0.92, 0.7]  # Different heights for the brackets above the boxplot
        bracket_positions = [(1, 2), (3, 4)]  # (start, end) of the brackets
        p_values = [ttest_pvalue_individual_conventional, ttest_pvalue_individual_conventional_dh]
        if significnce_brackets == 'stars':
            bracket_heights = [0.92, 0.7]
            bracket_positions = [(1, 2), (3, 4)]
            self.plot_significance_stars(ax, bracket_positions, p_values, bracket_heights, position="below")
        else:
            self.plot_significance_brackets(ax, bracket_positions, p_values, bracket_heights, position="above")

        plt.savefig(os.path.join(save_path.downloadsPath, 'Youden_Index.pdf'), bbox_inches='tight')
        plt.show()

    def order_df_by_FMA(self, df): #TODO: coppied from primitive_analysis.py should be moved to utilities.py
        """
        Orders the given DataFrame by the 'FMA_UE' column and returns the sorted DataFrame along with the 'participantID',
        'FMA_UE', and 'ARAT' columns as separate variables. If the 'FMA_UE' column contains the same values for multiple rows, the rows are sorted by the 'ARAT' column.
        Parameters:
            df (pandas.DataFrame): The DataFrame to be sorted.
        Returns:
            tuple: A tuple containing the sorted DataFrame, 'participantID' labels, 'FMA_UE' labels, and 'ARAT' labels.
        """
        df['FMA_UE'] = self.FMA_UE
        df['ARAT'] = self.ARAT
        df.sort_values(['FMA_UE', 'ARAT'], inplace=True)
        ID_labels = df['participantID']
        FMA_labels = df.pop('FMA_UE')
        ARAT_labels = df.pop('ARAT')
        return df, ID_labels, FMA_labels, ARAT_labels

    def AUC_performance_per_subject(self, side='NDH'):
        colors = thesis_style.get_thesis_colours()
        if side == 'NDH':
            data = {
                'participantID': self.PARTICIPANT_ID,
                'Conventional thresholds': self.conventional_AUC_list_ndh,
                'Individual thresholds': self.individual_AUC_list_ndh
            }
            df = pd.DataFrame(data)
            title_side = 'affected side'
            color = [colors['affected'], colors['light_orange']]
        elif side == 'DH':
            data = {
                'participantID': self.PARTICIPANT_ID,
                'Conventional thresholds': self.conventional_AUC_list_dh,
                'Individual thresholds': self.individual_AUC_list_dh
            }
            df = pd.DataFrame(data)
            title_side = 'unaffected side'
            color = [colors['conventional'], colors['individual']]
        else:
            raise ValueError('side must be either "NDH" or "DH"')

        df_ordered, ID_label, FMA_label, ARAT_label = self.order_df_by_FMA(df.copy())

        fig, ax = plt.subplots()
        # AUC is clinically useful (≥0.75) according to [Fan et al., 2006]
        ax.axhline(y=0.75, color=colors['grey'], linestyle='dotted', label='Clinically required performance [Fan et al., 2006]', lw=2.0)
        # random classifier
        ax.axhline(y=0.5, color=colors['black_grey'], linestyle='--', label='Random classifier', lw=1.3)
    
        df_ordered.plot(ax=ax, x='participantID', y='Conventional thresholds', kind='line', legend=True, color=color[0], marker='s', markersize=7, linewidth=1, linestyle='-.')
        df_ordered.plot(ax=ax, x='participantID', y='Individual thresholds', kind='line', legend=True, color=color[1], marker='o', markersize=8, linewidth=1, linestyle='-.')
    
        plt.ylabel('ROC AUC', fontsize=10)
        plt.yticks(fontsize=10)
        ax.set_ylim(0.45, 1.0)
        plt.xlabel('')
        plt.xticks(range(len(self.PARTICIPANT_ID)),
                [f"{id_conversion.get_thesisID(id)}\nFMA-UE: {int(fma)}\nARAT: {int(arat)}" for id, fma, arat in zip(ID_label, FMA_label, ARAT_label)],
                rotation=0, fontsize=8)
        plt.tight_layout(rect=[0, 0, 1.4, 1])

        # Adding brackets and labels for FMA-UE categories (Woytowicz et al., 2017)
        # attention: bracket possitions are hardcoded
        ax.annotate('', xy=(0.299, -0.14), xytext=(0, -0.14), xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-', lw=1.0, color='black'))
        ax.annotate('severe impairment', xy=(0.15, -0.18), xycoords='axes fraction', ha='center', fontsize=8)

        ax.annotate('', xy=(0.499, -0.14), xytext=(0.301, -0.14), xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-', lw=1.0, color='black'))
        ax.annotate('moderate impairment', xy=(0.4, -0.18), xycoords='axes fraction', ha='center', fontsize=8)

        ax.annotate('', xy=(1.0, -0.14), xytext=(0.501, -0.14), xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-', lw=1.0, color='black'))
        ax.annotate('mild impairment', xy=(0.75, -0.18), xycoords='axes fraction', ha='center', fontsize=8)

        plt.legend(loc='upper left', frameon=False, fontsize=10)
        plt.title('Threshold performance ' + title_side)
        plt.savefig(os.path.join(save_path.downloadsPath, f'AUC_performance_{side}.pdf'), bbox_inches='tight')
        plt.show()