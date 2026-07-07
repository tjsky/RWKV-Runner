package backend_golang

import (
	"errors"
	"strconv"
	"strings"
)

func (a *App) GetNvidiaGpuCount() (int, error) {
	// temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used
	// gpu_name,gpu_bus_id,driver_version
	// nvidia-smi --help-query-gpu
	output, err := a.CommandOutput("nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits")
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(output)
}

func (a *App) GetCudaComputeCapability(index int) (string, error) {
	output, err := a.CommandOutput("nvidia-smi", "-i="+strconv.Itoa(index), "--query-gpu=compute_cap", "--format=csv,noheader,nounits")
	if err != nil {
		return "", err
	}

	if output == "" {
		return "", errors.New("compute capability is empty")
	}

	return output, nil
}

func (a *App) GetMaxCudaComputeCapability() (string, error) {
	gpuCount, err := a.GetNvidiaGpuCount()
	if err != nil {
		return "", err
	}
	maxComputeCap := "0.0"
	for i := 0; i < gpuCount; i++ {
		computeCap, err := a.GetCudaComputeCapability(i)
		if err != nil {
			return "", err
		}
		computeCapFloat, err := strconv.ParseFloat(computeCap, 64)
		if err != nil {
			return "", err
		}
		maxComputeCapFloat, err := strconv.ParseFloat(maxComputeCap, 64)
		if err != nil {
			return "", err
		}
		if computeCapFloat > maxComputeCapFloat {
			maxComputeCap = computeCap
		}
	}
	if maxComputeCap == "0.0" {
		return "", errors.New("no cuda compute capability")
	}
	return maxComputeCap, nil
}

func (a *App) GetSupportedCudaVersion() (string, error) {
	output, err := a.CommandOutput("nvidia-smi", "--query")
	if err != nil {
		return "", err
	}

	return parseSupportedCudaVersion(output)
}

func parseSupportedCudaVersion(output string) (string, error) {
	var fallbackVersion string

	for _, line := range strings.Split(output, "\n") {
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}

		version := parseCudaVersion(value)
		if version == "" {
			continue
		}

		switch strings.TrimSpace(key) {
		case "CUDA UMD Version":
			return version, nil
		case "CUDA Version":
			if fallbackVersion == "" {
				fallbackVersion = version
			}
		}
	}

	if fallbackVersion != "" {
		return fallbackVersion, nil
	}
	return "", errors.New("cuda version is empty")
}

func parseCudaVersion(value string) string {
	fields := strings.Fields(value)
	if len(fields) == 0 {
		return ""
	}

	version := fields[0]
	parts := strings.Split(version, ".")
	if len(parts) < 2 {
		return ""
	}
	for _, part := range parts {
		if _, err := strconv.Atoi(part); err != nil {
			return ""
		}
	}
	return version
}

func (a *App) GetTorchVersion(python string) (string, error) {
	var err error
	if python == "" {
		python, err = a.GetPython()
		if err != nil {
			return "", err
		}
	}

	output, err := a.CommandOutput(python, "-c", "import torch; print(torch.__version__)")
	if err != nil {
		return "", err
	}

	if output == "" {
		return "", errors.New("torch version is empty")
	}

	return output, nil
}
