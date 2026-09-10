<template>
  <section class="plugin-token-card" :aria-label="t('pluginRanking.title', { range: rangeLabel })">
    <h2>{{ t('pluginRanking.title', { range: rangeLabel }) }}</h2>
    <p class="coverage-note">{{ t('pluginRanking.coverage') }}</p>
    <p class="coverage-note">{{ t('pluginRanking.overlap') }}</p>
    <v-progress-linear v-if="loading" indeterminate class="mt-4" />
    <v-alert v-else-if="error" type="error" variant="tonal" class="mt-4">{{ error }}</v-alert>
    <div v-else-if="items.length" class="table-scroll">
      <table>
        <thead>
          <tr>
            <th scope="col">{{ t('pluginRanking.plugin') }}</th>
            <th v-for="column in columns" :key="column" scope="col" class="token-value">
              {{ t(`pluginRanking.${column}`) }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in items" :key="item.plugin_id">
            <th scope="row" class="plugin-name">
              {{ item.display_name }}
              <span v-if="item.display_name !== item.plugin_id" class="plugin-id">{{ item.plugin_id }}</span>
            </th>
            <td v-for="column in columns" :key="column" class="token-value">
              {{ formatter.format(item[column]) }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <p v-else class="empty-state">{{ t('pluginRanking.empty') }}</p>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { PluginTokenStatsData } from '@/api/generated/openapi-v1'
import { useI18n, useModuleI18n } from '@/i18n/composables'

defineProps<{
  items: PluginTokenStatsData['items']
  rangeLabel: string
  loading: boolean
  error: string
}>()

const { locale } = useI18n()
const { tm: t } = useModuleI18n('features/stats')
const formatter = computed(() => new Intl.NumberFormat(locale.value))
const columns = ['token_input_other', 'token_input_cached', 'token_output', 'total_tokens'] as const
</script>

<style scoped>
.plugin-token-card {
  margin-top: 20px;
  padding: 22px;
  border-radius: 16px;
  background: var(--stats-card);
}

h2 {
  font-size: 17px;
  font-weight: 600;
}

.coverage-note {
  margin-top: 8px;
  color: var(--stats-muted);
  font-size: 12px;
}

.table-scroll {
  overflow-x: auto;
  margin-top: 16px;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

th, td {
  padding: 12px 8px;
  border-bottom: 1px solid var(--stats-border);
  text-align: left;
}

thead th {
  color: var(--stats-muted);
  font-weight: 500;
}

tbody tr:last-child > * {
  border-bottom: 0;
}

.token-value {
  text-align: right;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}

.plugin-name {
  min-width: 140px;
  overflow-wrap: anywhere;
  font-weight: 500;
}

.plugin-id {
  display: block;
  color: var(--stats-subtle);
  font-size: 12px;
}

.empty-state {
  margin-top: 20px;
  color: var(--stats-muted);
  font-size: 14px;
}
</style>
