import { useMemo } from 'react'
import { Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Dataset, ExperimentSummary, PointSeries } from '../lib/api'
import { API_URL } from '../lib/api'
import { formatMetric, formatPercent } from '../lib/format'

type ChartPoint = {
  time: string
  original: number
  masked: number | null
  imputed: number
  label: string
  tooltipLabel: string
}

type Props = {
  experimentSummary: ExperimentSummary
  selectedDataset?: Dataset
  normalizedPoints: {
    point_id: string
    label: string
    lat: number
    lon: number
    region: string
    missing_ratio: number
    metrics: { mae: number; mse: number; rmse: number }
  }[]
  selectedPointId: string | null
  onSelectPoint: (pointId: string) => void
  pointSeries: PointSeries | null
  loadingPoint: boolean
  timelineData: { label: string; overall: number; selected: number }[]
  filteredChartData: ChartPoint[]
  currentTab: 'info' | 'spatial' | 'metrics'
  onChangeTab: (tab: 'info' | 'spatial' | 'metrics') => void
  availableMonths: { key: string; label: string }[]
  selectedMonth: string | null
  onChangeMonth: (month: string) => void
}

const ResultsView = ({
  experimentSummary,
  selectedDataset,
  normalizedPoints,
  selectedPointId,
  onSelectPoint,
  pointSeries,
  loadingPoint,
  timelineData,
  filteredChartData,
  currentTab,
  onChangeTab,
  availableMonths,
  selectedMonth,
  onChangeMonth,
}: Props) => {
  const mapUrl = useMemo(() => {
    if (!selectedDataset) return ''
    const params = new URLSearchParams()
    if (selectedPointId) {
      params.set('highlight_point', selectedPointId)
    }
    const query = params.toString()
    return `${API_URL}/datasets/${selectedDataset.id}/map${query ? `?${query}` : ''}`
  }, [selectedDataset?.id, selectedPointId])

  const seriesLabel = (key: string) => {
    if (key === 'original') return 'Observado'
    if (key === 'masked') return 'Com falhas'
    if (key === 'imputed') return 'Imputado'
    return key
  }

  const tooltipLabelFormatter = (label: any, payload: any) => payload?.[0]?.payload?.tooltipLabel ?? label

  const tooltipValueFormatter = (value: any, name: string) => {
    if (value === null || value === undefined || value === '') {
      return ['—', seriesLabel(name)]
    }
    if (typeof value === 'number') {
      return [value.toFixed(3), seriesLabel(name)]
    }
    return [value, seriesLabel(name)]
  }

  const seriesConfigs = [
    { key: 'original', title: 'Série observada', color: '#2563eb', background: '' },
    { key: 'masked', title: 'Dados faltantes (máscara)', color: '#f97316', background: 'warn-bg' },
    { key: 'imputed', title: 'Série imputada', color: '#22c55e', background: 'success-bg' },
  ] as const

  const variableLabel =
    selectedDataset?.variables.find((variable) => variable.id === experimentSummary.dataset.variable)?.label ??
    experimentSummary.dataset.variable

  return (
    <section className="results-card">
      <div className="tabs">
      <button className={currentTab === 'info' ? 'active' : ''} onClick={() => onChangeTab('info')}>
        Informações Gerais
      </button>
      <button className={currentTab === 'spatial' ? 'active' : ''} onClick={() => onChangeTab('spatial')}>
        Análise Espaço-Temporal
      </button>
      <button className={currentTab === 'metrics' ? 'active' : ''} onClick={() => onChangeTab('metrics')}>
        Métricas de Desempenho
      </button>
      </div>

      {currentTab === 'info' && (
      <div className="info-grid">
        <div className="info-card">
          <h4>Informações do conjunto</h4>
          <div className="info-row">
            <span>Nome</span>
            <strong>{experimentSummary.dataset.name}</strong>
          </div>
          <div className="info-row">
            <span>Variável</span>
            <strong>{variableLabel}</strong>
          </div>
          <div className="info-row">
            <span>Período</span>
            <strong>{experimentSummary.dataset.statistics.time_span}</strong>
          </div>
          <div className="info-row">
            <span>Frequência</span>
            <strong>{experimentSummary.dataset.statistics.resolution}</strong>
          </div>
        </div>
        <div className="info-card">
          <h4>Estatísticas espaciais</h4>
          <div className="info-row">
            <span>Número de estações</span>
            <strong>{experimentSummary.dataset.statistics.points}</strong>
          </div>
          <div className="info-row">
            <span>Média de gaps</span>
            <strong>{experimentSummary.dataset.statistics.avg_gap_hours}h</strong>
          </div>
          <div className="info-row">
            <span>Falhas nativas</span>
            <strong>{formatPercent(experimentSummary.dataset.statistics.native_missing_ratio)}</strong>
          </div>
        </div>
        <div className="info-card">
          <h4>Configurações de imputação</h4>
          <div className="info-row">
            <span>Método</span>
            <strong>{experimentSummary.method.name}</strong>
          </div>
          <div className="info-row">
            <span>Categoria</span>
            <strong>{experimentSummary.method.category}</strong>
          </div>
        </div>
      </div>
    )}

    {currentTab === 'spatial' && (
      <div className="spatial-grid">
        <div className="map-card">
          <div className="label-line">
            <h4>Mapa fixo das estações</h4>
          </div>
          {selectedDataset && mapUrl && (
            <div className="map-wrapper extra-tall">
              <img src={mapUrl} alt="Mapa das estações" className="map-image" />
            </div>
          )}
          <div className="map-list-panel full">
            <div className="map-list-header">
              <div>
                <h5>Lista de estações</h5>
                <p className="muted tiny">Clique para selecionar e ver a série temporal</p>
              </div>
            </div>
            <div className="map-station-list">
              {normalizedPoints.map((point) => (
                <button
                  key={point.point_id}
                  className={`map-list-item ${selectedPointId === point.point_id ? 'active' : ''}`}
                  onClick={() => onSelectPoint(point.point_id)}
                >
                  <div className="map-list-title">
                    <strong>{point.label}</strong>
                    <span className="pill small">{formatPercent(point.missing_ratio)} falta</span>
                  </div>
                  <p className="muted tiny">{point.region}</p>
                  <div className="map-list-meta">
                    <span>RMSE {formatMetric(point.metrics.rmse)}</span>
                    <span>MAE {formatMetric(point.metrics.mae)}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="series-card">
          {pointSeries ? (
            <>
              <div className="series-header stacked">
                <div>
                  <h4>Séries temporais imputadas</h4>
                  <p className="muted small">Visualização vertical com rótulos de data e valor em português.</p>
                </div>
                <div className="series-actions">
                  <label className="muted small" htmlFor="monthSelector">
                    Exibir mês
                  </label>
                  <select
                    id="monthSelector"
                    value={selectedMonth ?? availableMonths[0]?.key ?? ''}
                    onChange={(event) => onChangeMonth(event.target.value)}
                    disabled={!availableMonths.length}
                  >
                    {availableMonths.map((month) => (
                      <option key={month.key} value={month.key}>
                        {month.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="series-meta-row">
                <div>
                  <h5>{pointSeries.label}</h5>
                  <p className="muted small">
                    Lat {pointSeries.lat.toFixed(3)}, Lon {pointSeries.lon.toFixed(3)}
                  </p>
                </div>
                {loadingPoint && <span className="pill">Carregando série…</span>}
              </div>
              <div className="series-stack">
                {seriesConfigs.map((serie) => (
                  <div key={serie.key} className={`mini-chart stacked-chart ${serie.background}`}>
                    <div className="mini-title">{serie.title}</div>
                    <ResponsiveContainer width="100%" height={170}>
                      <LineChart data={filteredChartData} margin={{ top: 6, right: 8, bottom: 6, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="label" tick={{ fontSize: 12 }} minTickGap={18} />
                        <YAxis tick={{ fontSize: 12 }} width={40} />
                        <Tooltip labelFormatter={tooltipLabelFormatter} formatter={(value, name) => tooltipValueFormatter(value, name as string)} />
                        <Line type="monotone" dataKey={serie.key} stroke={serie.color} dot={false} strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ))}
              </div>
              <div className="info-row metrics-row">
                <div>
                  <span>MAE</span>
                  <strong>{formatMetric(pointSeries.metrics.mae)}</strong>
                </div>
                <div>
                  <span>MSE</span>
                  <strong>{formatMetric(pointSeries.metrics.mse)}</strong>
                </div>
                <div>
                  <span>RMSE</span>
                  <strong>{formatMetric(pointSeries.metrics.rmse)}</strong>
                </div>
              </div>
            </>
          ) : (
            <p className="muted">Selecione uma estação na lista para visualizar as séries.</p>
          )}
        </div>
      </div>
    )}

    {currentTab === 'metrics' && (
      <div className="metrics-view">
        <div className="highlight-metrics">
          <div className="metric-tile success-bg">
            <div>MAE geral do conjunto</div>
            <h2>{experimentSummary.metrics_overview.mae.toFixed(6)}</h2>
            <p className="muted small">Erro absoluto médio</p>
          </div>
          <div className="metric-tile warn-bg">
            <div>MSE geral do conjunto</div>
            <h2>{experimentSummary.metrics_overview.mse.toFixed(6)}</h2>
            <p className="muted small">Erro quadrático médio</p>
          </div>
        </div>
        <div className="timeline-card">
          <h4>Linha do tempo de falhas</h4>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={timelineData}>
              <defs>
                <linearGradient id="overall" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#2563eb" stopOpacity={0.7} />
                  <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="selected" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f97316" stopOpacity={0.7} />
                  <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" />
              <YAxis tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
              <Tooltip formatter={(value: number) => formatPercent(value)} />
              <Area type="monotone" dataKey="overall" stroke="#2563eb" fillOpacity={1} fill="url(#overall)" name="Geral" />
              <Area type="monotone" dataKey="selected" stroke="#f97316" fillOpacity={1} fill="url(#selected)" name="Selecionados" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="table-card">
          <div className="table-header">
            <h4>Métricas por Estação</h4>
            <span className="muted small">Clique para focar na estação</span>
          </div>
          <div className="table">
            <div className="table-row head">
              <span>Estação</span>
              <span>MAE</span>
              <span>MSE</span>
              <span>RMSE</span>
              <span>Faltantes</span>
            </div>
            {experimentSummary.points_overview.map((point) => (
              <button
                key={point.point_id}
                className={`table-row ${selectedPointId === point.point_id ? 'active' : ''}`}
                onClick={() => onSelectPoint(point.point_id)}
              >
                <span>{point.label}</span>
                <span>{formatMetric(point.metrics.mae)}</span>
                <span>{formatMetric(point.metrics.mse)}</span>
                <span>{formatMetric(point.metrics.rmse)}</span>
                <span>{formatPercent(point.missing_ratio)}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    )}
    </section>
  )
}

export default ResultsView
