import type { Dataset } from '../lib/api'
import { formatPercent } from '../lib/format'

type Props = {
  datasets: Dataset[]
  selectedDatasetId?: string
  selectedVariable?: string
  onSelectDataset: (datasetId: string) => void
  onSelectVariable: (variableId: string) => void
}

const DatasetStep = ({ datasets, selectedDatasetId, selectedVariable, onSelectDataset, onSelectVariable }: Props) => {
  return (
    <div className="section">
      <div className="section-header align-center">
        <div>
          <h2>Selecione o conjunto de dados</h2>
          <p>Visualize os experimentos pré-processados para ROMS (SSH) e AQI-36 (PM2.5).</p>
        </div>
      </div>
      <div className="card-grid two-col">
        {datasets.map((dataset) => (
          <button
            key={dataset.id}
            className={`select-card ${dataset.id === selectedDatasetId ? 'active' : ''}`}
            onClick={() => onSelectDataset(dataset.id)}
          >
            <div className="card-title-line">
              <h3>{dataset.name}</h3>
              {dataset.id === selectedDatasetId && <span className="checkmark">✓</span>}
            </div>
            <p className="muted">{dataset.description}</p>
            <ul className="bullet-list">
              <li>{dataset.statistics.resolution}</li>
              <li>{dataset.variables.length} variáveis</li>
              <li>Falha nativa {formatPercent(dataset.statistics.native_missing_ratio)}</li>
              {dataset.available_missing_ratios && dataset.available_missing_ratios.length > 0 && (
                <li>
                  Percentuais pré-processados:{' '}
                  {dataset.available_missing_ratios.map((value) => `${(value * 100).toFixed(0)}%`).join(', ')}
                </li>
              )}
            </ul>
          </button>
        ))}
      </div>
      {selectedDatasetId && (
        <div className="form-row">
          <label>Variável</label>
          <select value={selectedVariable} onChange={(event) => onSelectVariable(event.target.value)}>
            {datasets
              .find((dataset) => dataset.id === selectedDatasetId)
              ?.variables.map((variable) => (
                <option key={variable.id} value={variable.id}>
                  {variable.label} · {variable.units}
                </option>
              ))}
          </select>
        </div>
      )}
    </div>
  )
}

export default DatasetStep
