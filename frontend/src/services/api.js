import axios from 'axios'

const api = axios.create({
  baseURL: '',
  timeout: 120000,
})

export async function checkHealth() {
  const { data } = await api.get('/health')
  return data
}

export async function analyzeImage(file) {
  const formData = new FormData()

  formData.append('file', file)

  const { data } = await api.post(
    '/analyze/image',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  )

  return data
}

export default api