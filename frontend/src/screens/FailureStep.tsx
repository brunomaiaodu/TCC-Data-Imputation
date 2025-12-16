import { useMemo } from 'react'
import type { Dataset } from '../lib/api'
import { API_URL } from '../lib/api'
import { formatPercent } from '../lib/format'

type Props = {
  dataset?: Dataset
  missingMode: 'general' | 'selected' | 'original'
  missingRatio: number
  selectedPoints: string[]
  onModeChange: (mode: 'general' | 'selected' | 'original') => void
  onMissingRatioChange: (ratio: number) => void
  onTogglePoint: (pointId: string) => void
  onPointRatioChange: (pointId: string, value: number) => void
  stationRatios: Record<string, number>
  errorMessage?: string | null
}

const FailureStep = ({
  dataset,
  missingMode,
  missingRatio,
  selectedPoints,
  onModeChange,
  onMissingRatioChange,
  onTogglePoint,
  onPointRatioChange,
  stationRatios,
  errorMessage,
}: Props) => {
  const ratioOptions = useMemo(() => {
    if (!dataset?.available_missing_ratios || !dataset.available_missing_ratios.length) {
      return null
    }
    return dataset.available_missing_ratios
  }, [dataset?.available_missing_ratios])
  const mapUrl = useMemo(() => {
    if (!dataset) return ''
    const selectedParam = selectedPoints.length ? `selected_points=${selectedPoints.join(',')}` : ''
    const query = [selectedParam].filter(Boolean).join('&')
    return `${API_URL}/datasets/${dataset.id}/map${query ? `?${query}` : ''}`
  }, [dataset, selectedPoints])
  return (
    <div className="section">
      <div className="section-header align-center">
        <div>
          <h2>Configuração de Falhas</h2>
          <p>Use o modo de falha disponível para cada conjunto (ROMS: percentuais fixos · AQI-36: falha original).</p>
        </div>
      </div>
      <div className="failure-column">
        {dataset?.failure_modes.map((mode) => (
          <button
            key={mode.id}
            className={`select-card stacked ${missingMode === mode.id ? 'active' : ''}`}
            onClick={() => onModeChange(mode.id)}
          >
            <div className="card-title-line">
              <h3>{mode.name}</h3>
              {missingMode === mode.id && <span className="checkmark">✓</span>}
            </div>
            <p className="muted small">{mode.description}</p>
            {mode.id !== 'selected' && (
              <div className="pill-row">
                <span className="pill">Recomendado {formatPercent(mode.recommended_ratio)}</span>
              </div>
            )}
            {missingMode === mode.id && mode.id === 'original' && (
              <div className="expand-panel">
                <strong>Percentual detectado</strong>
                {dataset ? (
                  <p>
                    Total de pontos com falta: {formatPercent(dataset.statistics.native_missing_ratio)} em {dataset.statistics.points} estações.
                  </p>
                ) : (
                  <p>Carregue um dataset para visualizar.</p>
                )}
              </div>
            )}
            {missingMode === mode.id && mode.id === 'general' && (
              <div className="expand-panel">
                <label className="muted small">Percentual pré-processado (%)</label>
                {ratioOptions ? (
                  <div className="pill-row wrap">
                    {ratioOptions.map((value) => (
                      <button
                        key={value}
                        type="button"
                        className={`chip ${missingRatio === value ? 'active' : ''}`}
                        onClick={(event) => {
                          event.stopPropagation()
                          onMissingRatioChange(value)
                        }}
                      >
                        {(value * 100).toFixed(0)}%
                      </button>
                    ))}
                  </div>
                ) : (
                  <input
                    id="generalRatio"
                    type="number"
                    min={0}
                    max={80}
                    value={(missingRatio * 100).toFixed(1)}
                    onChange={(event) => onMissingRatioChange(Math.min(0.8, Math.max(0, Number(event.target.value) / 100)))}
                  />
                )}
              </div>
            )}
            {missingMode === mode.id && mode.id === 'selected' && (
              <div className="expand-panel">
                <p className="muted small">Configurar percentual para cada estação.</p>
                <div className="custom-grid">
                  <div className="map-panel">
                    {dataset && mapUrl && <img src={mapUrl} alt="Mapa do dataset" className="map-image" />}
                  </div>
                  <div className="station-list">
                    {dataset?.points.map((point) => (
                      <label key={point.id} className="station-row">
                        <div>
                          <strong>{point.label}</strong>
                          <p className="muted tiny">{point.region}</p>
                        </div>
                        <div className="station-input">
                          <input
                            type="number"
                            min={0}
                            max={80}
                            value={((stationRatios[point.id] ?? dataset?.statistics.native_missing_ratio ?? 0) * 100).toFixed(1)}
                            onChange={(event) => onPointRatioChange(point.id, Math.min(0.8, Math.max(0, Number(event.target.value) / 100)))}
                          />
                          <span>%</span>
                        </div>
                        <button type="button" className={`chip ${selectedPoints.includes(point.id) ? 'active' : ''}`} onClick={() => onTogglePoint(point.id)}>
                          {selectedPoints.includes(point.id) ? 'Selecionada' : 'Selecionar'}
                        </button>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </button>
        ))}
      </div>
      {errorMessage && <div className="error-banner">{errorMessage}</div>}
    </div>
  )
}

export default FailureStep
