import type { Method } from '../lib/api'

type Props = {
  methods: Method[]
  selectedMethodId?: string
  onSelectMethod: (methodId: string) => void
}

const MethodStep = ({ methods, selectedMethodId, onSelectMethod }: Props) => {
  return (
    <div className="section">
      <div className="section-header align-center">
        <div>
          <h2>Métodos de Imputação</h2>
          <p>Selecione o método que deseja visualizar (SPIN ou KNN) para comparar os resultados.</p>
        </div>
      </div>
      <div className="card-grid three-col">
        {methods.map((method) => (
          <button
            key={method.id}
            className={`select-card ${method.id === selectedMethodId ? 'active' : ''}`}
            onClick={() => onSelectMethod(method.id)}
          >
            <div className="card-title-line">
              <h3>{method.name}</h3>
              {method.id === selectedMethodId && <span className="checkmark">✓</span>}
            </div>
            <p className="muted">{method.category}</p>
            <div className="pill-row">
              <span className="pill success">Ponto forte · {method.strengths[0]}</span>
              <span className="pill warn">Limitação · {method.limitations[0]}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

export default MethodStep
