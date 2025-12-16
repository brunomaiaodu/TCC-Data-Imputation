export type DatasetVariable = {
  id: string
  label: string
  units?: string
}

export type DatasetPoint = {
  id: string
  label: string
  lat: number
  lon: number
  dataset_index: number
  region: string
}

export type FailureMode = {
  id: 'general' | 'selected' | 'original'
  name: string
  description: string
  recommended_ratio: number
  block_range: [number, number]
}

export type DatasetStatistics = {
  points: number
  variables: number
  native_missing_ratio: number
  avg_gap_hours: number
  time_span: string
  resolution: string
}

export type Dataset = {
  id: string
  name: string
  description: string
  variables: DatasetVariable[]
  default_variable: string
  statistics: DatasetStatistics
  map_center: { lat: number; lon: number; zoom: number }
  color: string
  available_missing_ratios?: number[]
  failure_modes: FailureMode[]
  points: DatasetPoint[]
}

export type Method = {
  id: string
  name: string
  category: string
  latency: string
  strengths: string[]
  limitations: string[]
}

export type MissingConfig = {
  mode: 'general' | 'selected' | 'original'
  ratio: number
  block_length: number
  selected_points: string[]
  station_ratios?: Record<string, number>
}

export type ExperimentSummary = {
  experiment_id: string
  dataset: {
    id: string
    name: string
    variable: string
    color: string
    statistics: DatasetStatistics
  }
  method: Method
  missing_config: MissingConfig
  missing_stats: {
    overall_missing_ratio: number
    selected_missing_ratio: number
    original_missing_ratio: number
    masked_points: number
    total_points: number
    timeline: { label: string; overall: number; selected: number }[]
  }
  points_overview: {
    point_id: string
    label: string
    lat: number
    lon: number
    region: string
    missing_ratio: number
    metrics: { mae: number; mse: number; rmse: number }
  }[]
  metrics_overview: { mae: number; mse: number; rmse: number }
  generated_insights: { title: string; details: string }[]
  created_at: string
}

export type PointSeries = {
  point_id: string
  label: string
  time: string[]
  original: number[]
  masked: Array<number | null>
  imputed: number[]
  metrics: { mae: number; mse: number; rmse: number }
  lat: number
  lon: number
}

export const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options?.headers ?? {}) },
    ...options,
  })
  if (!response.ok) {
    const message = await response.text()
    throw new Error(message || 'Request failed')
  }
  return response.json() as Promise<T>
}

export function fetchDatasets(): Promise<Dataset[]> {
  return request<Dataset[]>('/datasets')
}

export function fetchMethods(): Promise<Method[]> {
  return request<Method[]>('/methods')
}

export function createExperiment(payload: {
  dataset_id: string
  variable: string
  method_id: string
  missing_config: MissingConfig
}): Promise<ExperimentSummary> {
  return request<ExperimentSummary>('/experiments', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function fetchPointSeries(experimentId: string, pointId: string): Promise<PointSeries> {
  return request<PointSeries>(`/experiments/${experimentId}/points/${pointId}`)
}
