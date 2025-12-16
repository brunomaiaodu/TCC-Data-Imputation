import { useEffect, useMemo, useState } from 'react'
import type { Dataset, ExperimentSummary, Method, PointSeries } from './lib/api'
import { createExperiment, fetchDatasets, fetchMethods, fetchPointSeries } from './lib/api'
import DatasetStep from './screens/DatasetStep'
import FailureStep from './screens/FailureStep'
import MethodStep from './screens/MethodStep'
import ResultsView from './screens/ResultsView'
import './App.css'

type Step = 1 | 2 | 3

function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [methods, setMethods] = useState<Method[]>([])
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>()
  const [selectedMethodId, setSelectedMethodId] = useState<string>()
  const [selectedVariable, setSelectedVariable] = useState<string>()
  const [missingMode, setMissingMode] = useState<'general' | 'selected' | 'original'>('general')
  const [missingRatio, setMissingRatio] = useState(0.2)
  const [blockLength, setBlockLength] = useState(24)
  const [selectedPoints, setSelectedPoints] = useState<string[]>([])
  const [experimentSummary, setExperimentSummary] = useState<ExperimentSummary | null>(null)
  const [selectedPointId, setSelectedPointId] = useState<string | null>(null)
  const [pointSeries, setPointSeries] = useState<PointSeries | null>(null)
  const [selectedMonth, setSelectedMonth] = useState<string | null>(null)
  const [initialLoading, setInitialLoading] = useState(true)
  const [runningExperiment, setRunningExperiment] = useState(false)
  const [loadingPoint, setLoadingPoint] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [step, setStep] = useState<Step>(1)
  const [resultsTab, setResultsTab] = useState<'info' | 'spatial' | 'metrics'>('info')
  const [customRatios, setCustomRatios] = useState<Record<string, number>>({})
  const resetExperimentState = () => {
    setExperimentSummary(null)
    setPointSeries(null)
    setSelectedPointId(null)
    setSelectedMonth(null)
    setResultsTab('info')
  }
  const goToStep = (target: Step) => {
    if (target < step && target <= 2) {
      resetExperimentState()
    }
    setStep(target)
  }
  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId),
    [datasets, selectedDatasetId],
  )
  const selectedMethod = useMemo(
    () => methods.find((method) => method.id === selectedMethodId),
    [methods, selectedMethodId],
  )

  useEffect(() => {
    async function bootstrap() {
      try {
        const [datasetResponse, methodResponse] = await Promise.all([fetchDatasets(), fetchMethods()])
        setDatasets(datasetResponse)
        setMethods(methodResponse)
        if (datasetResponse.length) {
          const firstDataset = datasetResponse[0]
          setSelectedDatasetId(firstDataset.id)
          setSelectedVariable(firstDataset.default_variable)
          const fallbackMode = firstDataset.failure_modes[0]?.id ?? 'general'
          setMissingMode(fallbackMode)
          const defaultRatio =
            firstDataset.available_missing_ratios?.[0] ?? firstDataset.failure_modes[0]?.recommended_ratio ?? 0.2
          setMissingRatio(defaultRatio)
          setBlockLength(
            Math.round(
              ((firstDataset.failure_modes[0]?.block_range[0] ?? 12) + (firstDataset.failure_modes[0]?.block_range[1] ?? 12)) /
                2,
            ),
          )
        }
        if (methodResponse.length) {
          setSelectedMethodId(methodResponse[0].id)
        }
      } catch (error) {
        setErrorMessage(error instanceof Error ? error.message : 'Não foi possível carregar os dados de configuração.')
      } finally {
        setInitialLoading(false)
      }
    }
    bootstrap()
  }, [])

  useEffect(() => {
    if (!selectedDataset) {
      return
    }

    setSelectedVariable(selectedDataset.default_variable)
    setSelectedPoints([])
    resetExperimentState()
    setStep(1)

    const mode = selectedDataset.failure_modes[0]?.id ?? 'general'
    setMissingMode(mode)
    const defaultRatio = selectedDataset.available_missing_ratios?.[0] ?? selectedDataset.failure_modes[0]?.recommended_ratio ?? 0.2
    setMissingRatio(defaultRatio)
    setBlockLength(
      Math.round(
        ((selectedDataset.failure_modes[0]?.block_range[0] ?? 12) + (selectedDataset.failure_modes[0]?.block_range[1] ?? 12)) / 2,
      ),
    )
    const seedRatios: Record<string, number> = {}
    selectedDataset.points.forEach((point) => {
      const baseRatio =
        customRatios[point.id] ??
        selectedDataset.statistics.native_missing_ratio ??
        selectedDataset.available_missing_ratios?.[0] ??
        0
      seedRatios[point.id] = baseRatio
    })
    setCustomRatios(seedRatios)
  }, [selectedDatasetId])

  useEffect(() => {
    if (!experimentSummary || !selectedPointId) {
      return
    }
    const experimentId = experimentSummary.experiment_id
    const pointId = selectedPointId!
    let cancelled = false
    async function fetchSeries() {
      try {
        setLoadingPoint(true)
        const series = await fetchPointSeries(experimentId, pointId)
        if (!cancelled) {
          setPointSeries(series)
        }
      } catch (error) {
        if (!cancelled) {
          setErrorMessage(error instanceof Error ? error.message : 'Falha ao carregar a série temporal.')
        }
      } finally {
        if (!cancelled) {
          setLoadingPoint(false)
        }
      }
    }
    fetchSeries()
    return () => {
      cancelled = true
    }
  }, [experimentSummary?.experiment_id, selectedPointId])

  const chartData = useMemo(() => {
    if (!pointSeries) {
      return []
    }
    const monthNames = [
      'janeiro',
      'fevereiro',
      'março',
      'abril',
      'maio',
      'junho',
      'julho',
      'agosto',
      'setembro',
      'outubro',
      'novembro',
      'dezembro',
    ]
    return pointSeries.time.map((timestamp, index) => {
      const [datePart, timePart] = timestamp.split(/[T ]/)
      const [year, monthRaw, dayRaw] = (datePart || '').split('-')
      const [hourRaw, minuteRaw] = (timePart || '').split(':')
      const month = monthRaw?.padStart(2, '0') || '01'
      const day = dayRaw?.padStart(2, '0') || '01'
      const hour = hourRaw?.padStart(2, '0') || '00'
      const minute = minuteRaw?.padStart(2, '0') || '00'
      const yearLabel = year || '0000'
      const monthLabel = `${monthNames[Number(month) - 1] ?? month} de ${yearLabel}`.trim()
      return {
        time: timestamp,
        original: pointSeries.original[index],
        masked: pointSeries.masked[index],
        imputed: pointSeries.imputed[index],
        label: `${day}/${month} ${hour}:${minute}`,
        tooltipLabel: `${day}/${month}/${year} ${hour}:${minute}`,
        monthKey: `${yearLabel}-${month}`,
        monthLabel,
      }
    })
  }, [pointSeries])

  const availableMonths = useMemo(() => {
    const monthMap = new Map<string, string>()
    chartData.forEach((entry) => {
      if (!monthMap.has(entry.monthKey)) {
        monthMap.set(entry.monthKey, entry.monthLabel)
      }
    })
    let months = Array.from(monthMap.entries()).map(([key, label]) => ({ key, label }))
    if (selectedDataset?.id === 'roms') {
      months = months.filter((month) => month.key.endsWith('-03') || month.key.endsWith('-06'))
    }
    return months
  }, [chartData, selectedDataset?.id])

  const filteredChartData = useMemo(() => {
    if (!selectedMonth) {
      return chartData
    }
    return chartData.filter((entry) => entry.monthKey === selectedMonth)
  }, [chartData, selectedMonth])

  useEffect(() => {
    if (!chartData.length) {
      setSelectedMonth(null)
      return
    }
    const monthExists = selectedMonth && chartData.some((entry) => entry.monthKey === selectedMonth)
    if (!monthExists) {
      setSelectedMonth(chartData[0].monthKey)
    }
  }, [chartData])

  const timelineData = experimentSummary?.missing_stats.timeline ?? []

  const normalizedPoints = useMemo(() => {
    if (experimentSummary) {
      return experimentSummary.points_overview
    }
    if (!selectedDataset) {
      return []
    }
    return selectedDataset.points.map((point) => ({
      point_id: point.id,
      label: point.label,
      lat: point.lat,
      lon: point.lon,
      region: point.region,
      missing_ratio: 0,
      metrics: { mae: 0, mse: 0, rmse: 0 },
    }))
  }, [experimentSummary, selectedDataset])

  const toggleTargetPoint = (pointId: string) => {
    setSelectedPoints((points) => (points.includes(pointId) ? points.filter((id) => id !== pointId) : [...points, pointId]))
  }

  const handlePointRatioChange = (pointId: string, value: number) => {
    setCustomRatios((previous) => ({ ...previous, [pointId]: value }))
  }

  const handleModeChange = (modeId: 'general' | 'selected' | 'original') => {
    setMissingMode(modeId)
    if (!selectedDataset) return
    const failure = selectedDataset.failure_modes.find((entry) => entry.id === modeId)
    if (failure) {
      const preset = selectedDataset.available_missing_ratios?.[0]
      setMissingRatio(preset ?? failure.recommended_ratio)
      setBlockLength(Math.round((failure.block_range[0] + failure.block_range[1]) / 2))
    }
    if (modeId !== 'selected') {
      setSelectedPoints([])
    }
    if (modeId === 'selected' && selectedDataset) {
      setSelectedPoints(selectedDataset.points.map((point) => point.id))
    }
  }

  useEffect(() => {
    if (experimentSummary && selectedMethodId) {
      resetExperimentState()
    }
  }, [selectedMethodId])

  const handleRunExperiment = async () => {
    if (!selectedDataset || !selectedMethod || !selectedVariable) {
      return
    }
    setRunningExperiment(true)
    setErrorMessage(null)
    try {
      const payload = {
        dataset_id: selectedDataset.id,
        variable: selectedVariable,
        method_id: selectedMethod.id,
        missing_config: {
          mode: missingMode,
          ratio: missingRatio,
          block_length: blockLength,
          selected_points: selectedPoints,
          station_ratios: customRatios,
        },
      }
      const result = await createExperiment(payload)
      setExperimentSummary(result)
      const defaultPoint = result.points_overview[0]
      setSelectedPointId(defaultPoint?.point_id ?? null)
      if (defaultPoint) {
        const series = await fetchPointSeries(result.experiment_id, defaultPoint.point_id)
        setPointSeries(series)
      } else {
        setPointSeries(null)
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Failed to create experiment.')
    } finally {
      setRunningExperiment(false)
    }
  }

  const stepIsComplete: Record<Step, boolean> = {
    1: Boolean(selectedDatasetId),
    2: Boolean(selectedMethodId),
    3: Boolean(selectedDatasetId && selectedMethodId && selectedVariable),
  }

  if (initialLoading) {
    return (
      <div className="app-shell">
        <header className="topbar">
          <div className="brand">Imputador Espaço-Temporal de Séries Temporais</div>
        </header>
        <div className="content">
          <div className="panel">
            <h2>Carregando configurações</h2>
            <p className="panel-subtitle">Buscando datasets e métodos...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">Imputador Espaço-Temporal</div>
      </header>
      <div className="content">
        <section className="wizard-card">
          <div className="stepper">
            {[1, 2, 3].map((value) => (
              <div key={value} className={`step ${step === value ? 'active' : ''} ${step > value ? 'done' : ''}`}>
                <span className="step-index">{value}</span>
                {value === 1 && <span className="step-label">Dataset</span>}
                {value === 2 && <span className="step-label">Método</span>}
                {value === 3 && <span className="step-label">Falhas</span>}
              </div>
            ))}
          </div>

          {step === 1 && (
            <DatasetStep
              datasets={datasets}
              selectedDatasetId={selectedDatasetId}
              selectedVariable={selectedVariable}
              onSelectDataset={setSelectedDatasetId}
              onSelectVariable={setSelectedVariable}
            />
          )}

          {step === 2 && (
            <MethodStep methods={methods} selectedMethodId={selectedMethodId} onSelectMethod={setSelectedMethodId} />
          )}

          {step === 3 && (
            <FailureStep
              dataset={selectedDataset}
              missingMode={missingMode}
              missingRatio={missingRatio}
              selectedPoints={selectedPoints}
              onModeChange={handleModeChange}
              onMissingRatioChange={setMissingRatio}
              onTogglePoint={toggleTargetPoint}
              onPointRatioChange={handlePointRatioChange}
              stationRatios={customRatios}
              errorMessage={errorMessage}
            />
          )}

          <div className="actions">
            <button className="ghost" disabled={step === 1} onClick={() => goToStep((step - 1) as Step)}>
              Voltar
            </button>
            {step < 3 && (
              <button
                className="primary"
                disabled={!stepIsComplete[step]}
                onClick={() => goToStep((step + 1) as Step)}
              >
                Próximo
              </button>
            )}
            {step === 3 && (
              <button className="primary" disabled={runningExperiment || !stepIsComplete[3]} onClick={handleRunExperiment}>
                {runningExperiment ? 'Gerando experimento…' : 'Avançar para Imputação'}
              </button>
            )}
          </div>
        </section>

        {experimentSummary && (
          <ResultsView
            experimentSummary={experimentSummary}
            selectedDataset={selectedDataset}
            normalizedPoints={normalizedPoints}
            selectedPointId={selectedPointId}
            onSelectPoint={setSelectedPointId}
            pointSeries={pointSeries}
            loadingPoint={loadingPoint}
            timelineData={timelineData}
            filteredChartData={filteredChartData}
            currentTab={resultsTab}
            onChangeTab={setResultsTab}
            availableMonths={availableMonths}
            selectedMonth={selectedMonth}
            onChangeMonth={setSelectedMonth}
          />
        )}
      </div>
    </div>
  )
}

export default App
