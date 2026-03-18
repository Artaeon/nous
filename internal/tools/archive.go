package tools

import (
	"archive/tar"
	"archive/zip"
	"compress/bzip2"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// FormatArchiveSize returns a human-readable size string.
func FormatArchiveSize(bytes int64) string {
	switch {
	case bytes >= 1<<30:
		return fmt.Sprintf("%.1fG", float64(bytes)/(1<<30))
	case bytes >= 1<<20:
		return fmt.Sprintf("%.1fM", float64(bytes)/(1<<20))
	case bytes >= 1<<10:
		return fmt.Sprintf("%.1fK", float64(bytes)/(1<<10))
	default:
		return fmt.Sprintf("%dB", bytes)
	}
}

// DetectArchiveFormat returns the archive format based on file extension.
func DetectArchiveFormat(path string) string {
	lower := strings.ToLower(path)
	switch {
	case strings.HasSuffix(lower, ".tar.gz") || strings.HasSuffix(lower, ".tgz"):
		return "tar.gz"
	case strings.HasSuffix(lower, ".tar.bz2") || strings.HasSuffix(lower, ".tbz2"):
		return "tar.bz2"
	case strings.HasSuffix(lower, ".tar"):
		return "tar"
	case strings.HasSuffix(lower, ".zip"):
		return "zip"
	default:
		return ""
	}
}

// ArchiveCompress creates an archive from the given path.
func ArchiveCompress(srcPath, outputPath, format string) (string, error) {
	srcPath, err := filepath.Abs(srcPath)
	if err != nil {
		return "", fmt.Errorf("archive: invalid path: %w", err)
	}

	info, err := os.Stat(srcPath)
	if err != nil {
		return "", fmt.Errorf("archive: source not found: %w", err)
	}

	if format == "" {
		format = "tar.gz"
	}

	if outputPath == "" {
		base := filepath.Base(srcPath)
		switch format {
		case "tar.gz":
			outputPath = base + ".tar.gz"
		case "zip":
			outputPath = base + ".zip"
		case "tar.bz2":
			outputPath = base + ".tar.bz2"
		case "tar":
			outputPath = base + ".tar"
		default:
			return "", fmt.Errorf("archive: unsupported format %q", format)
		}
		outputPath = filepath.Join(filepath.Dir(srcPath), outputPath)
	}

	switch format {
	case "tar.gz":
		err = compressTarGz(srcPath, outputPath, info)
	case "zip":
		err = compressZip(srcPath, outputPath, info)
	default:
		return "", fmt.Errorf("archive: unsupported format %q for compression", format)
	}
	if err != nil {
		return "", err
	}

	outInfo, err := os.Stat(outputPath)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("Created %s (%s)", outputPath, FormatArchiveSize(outInfo.Size())), nil
}

func compressTarGz(srcPath, outputPath string, srcInfo os.FileInfo) error {
	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("archive: cannot create output: %w", err)
	}
	defer outFile.Close()

	gzWriter := gzip.NewWriter(outFile)
	defer gzWriter.Close()

	tw := tar.NewWriter(gzWriter)
	defer tw.Close()

	basePath := filepath.Dir(srcPath)
	return filepath.Walk(srcPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(basePath, path)
		if err != nil {
			return err
		}

		header, err := tar.FileInfoHeader(info, "")
		if err != nil {
			return err
		}
		header.Name = relPath

		if err := tw.WriteHeader(header); err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()

		_, err = io.Copy(tw, f)
		return err
	})
}

func compressZip(srcPath, outputPath string, srcInfo os.FileInfo) error {
	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("archive: cannot create output: %w", err)
	}
	defer outFile.Close()

	zw := zip.NewWriter(outFile)
	defer zw.Close()

	basePath := filepath.Dir(srcPath)
	return filepath.Walk(srcPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(basePath, path)
		if err != nil {
			return err
		}

		if info.IsDir() {
			_, err := zw.Create(relPath + "/")
			return err
		}

		header, err := zip.FileInfoHeader(info)
		if err != nil {
			return err
		}
		header.Name = relPath
		header.Method = zip.Deflate

		w, err := zw.CreateHeader(header)
		if err != nil {
			return err
		}

		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()

		_, err = io.Copy(w, f)
		return err
	})
}

// ArchiveExtract extracts an archive to the given output directory.
func ArchiveExtract(archivePath, outputDir string) (string, error) {
	archivePath, err := filepath.Abs(archivePath)
	if err != nil {
		return "", fmt.Errorf("archive: invalid path: %w", err)
	}

	if _, err := os.Stat(archivePath); err != nil {
		return "", fmt.Errorf("archive: file not found: %w", err)
	}

	format := DetectArchiveFormat(archivePath)
	if format == "" {
		return "", fmt.Errorf("archive: cannot detect format for %q", archivePath)
	}

	if outputDir == "" {
		outputDir = filepath.Dir(archivePath)
	}

	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return "", fmt.Errorf("archive: cannot create output dir: %w", err)
	}

	var count int
	switch format {
	case "tar.gz":
		count, err = extractTarGz(archivePath, outputDir)
	case "tar.bz2":
		count, err = extractTarBz2(archivePath, outputDir)
	case "tar":
		count, err = extractTar(archivePath, outputDir)
	case "zip":
		count, err = extractZip(archivePath, outputDir)
	default:
		return "", fmt.Errorf("archive: unsupported format %q", format)
	}
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("Extracted %d items to %s", count, outputDir), nil
}

func extractTarGz(archivePath, outputDir string) (int, error) {
	f, err := os.Open(archivePath)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	gr, err := gzip.NewReader(f)
	if err != nil {
		return 0, fmt.Errorf("archive: gzip error: %w", err)
	}
	defer gr.Close()

	return extractFromTar(tar.NewReader(gr), outputDir)
}

func extractTarBz2(archivePath, outputDir string) (int, error) {
	f, err := os.Open(archivePath)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	br := bzip2.NewReader(f)
	return extractFromTar(tar.NewReader(br), outputDir)
}

func extractTar(archivePath, outputDir string) (int, error) {
	f, err := os.Open(archivePath)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	return extractFromTar(tar.NewReader(f), outputDir)
}

func extractFromTar(tr *tar.Reader, outputDir string) (int, error) {
	count := 0
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return count, fmt.Errorf("archive: tar read error: %w", err)
		}

		target := filepath.Join(outputDir, header.Name)

		// Prevent path traversal.
		if !strings.HasPrefix(filepath.Clean(target), filepath.Clean(outputDir)) {
			continue
		}

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0755); err != nil {
				return count, err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
				return count, err
			}
			outFile, err := os.Create(target)
			if err != nil {
				return count, err
			}
			if _, err := io.Copy(outFile, tr); err != nil {
				outFile.Close()
				return count, err
			}
			outFile.Close()
			count++
		}
	}
	return count, nil
}

func extractZip(archivePath, outputDir string) (int, error) {
	r, err := zip.OpenReader(archivePath)
	if err != nil {
		return 0, fmt.Errorf("archive: zip open error: %w", err)
	}
	defer r.Close()

	count := 0
	for _, f := range r.File {
		target := filepath.Join(outputDir, f.Name)

		// Prevent path traversal.
		if !strings.HasPrefix(filepath.Clean(target), filepath.Clean(outputDir)) {
			continue
		}

		if f.FileInfo().IsDir() {
			os.MkdirAll(target, 0755)
			continue
		}

		if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
			return count, err
		}

		rc, err := f.Open()
		if err != nil {
			return count, err
		}

		outFile, err := os.Create(target)
		if err != nil {
			rc.Close()
			return count, err
		}

		_, err = io.Copy(outFile, rc)
		outFile.Close()
		rc.Close()
		if err != nil {
			return count, err
		}
		count++
	}
	return count, nil
}

// ArchiveList lists the contents of an archive without extracting.
func ArchiveList(archivePath string) (string, error) {
	archivePath, err := filepath.Abs(archivePath)
	if err != nil {
		return "", fmt.Errorf("archive: invalid path: %w", err)
	}

	if _, err := os.Stat(archivePath); err != nil {
		return "", fmt.Errorf("archive: file not found: %w", err)
	}

	format := DetectArchiveFormat(archivePath)
	if format == "" {
		return "", fmt.Errorf("archive: cannot detect format for %q", archivePath)
	}

	var entries []string

	switch format {
	case "tar.gz":
		entries, err = listTarGz(archivePath)
	case "tar.bz2":
		entries, err = listTarBz2(archivePath)
	case "tar":
		entries, err = listTar(archivePath)
	case "zip":
		entries, err = listZip(archivePath)
	default:
		return "", fmt.Errorf("archive: unsupported format %q", format)
	}
	if err != nil {
		return "", err
	}

	if len(entries) == 0 {
		return "Archive is empty.", nil
	}

	return fmt.Sprintf("%d items:\n%s", len(entries), strings.Join(entries, "\n")), nil
}

func listTarGz(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gr, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gr.Close()

	return listFromTar(tar.NewReader(gr))
}

func listTarBz2(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return listFromTar(tar.NewReader(bzip2.NewReader(f)))
}

func listTar(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return listFromTar(tar.NewReader(f))
}

func listFromTar(tr *tar.Reader) ([]string, error) {
	var entries []string
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		sizeStr := FormatArchiveSize(header.Size)
		entries = append(entries, fmt.Sprintf("%8s  %s", sizeStr, header.Name))
	}
	return entries, nil
}

func listZip(path string) ([]string, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	var entries []string
	for _, f := range r.File {
		sizeStr := FormatArchiveSize(int64(f.UncompressedSize64))
		entries = append(entries, fmt.Sprintf("%8s  %s", sizeStr, f.Name))
	}
	return entries, nil
}

// RegisterArchiveTools adds the archive tool to the registry.
func RegisterArchiveTools(r *Registry) {
	r.Register(Tool{
		Name:        "archive",
		Description: "Compress and extract archives. Args: action (compress/extract/list), path (required), output (optional), format (tar.gz/zip/tar.bz2, auto-detect on extract).",
		Execute: func(args map[string]string) (string, error) {
			action := strings.ToLower(strings.TrimSpace(args["action"]))
			path := args["path"]
			output := args["output"]
			format := args["format"]

			if path == "" {
				return "", fmt.Errorf("archive: 'path' argument is required")
			}

			switch action {
			case "compress":
				return ArchiveCompress(path, output, format)
			case "extract":
				return ArchiveExtract(path, output)
			case "list":
				return ArchiveList(path)
			default:
				return "", fmt.Errorf("archive: unknown action %q (use compress, extract, or list)", action)
			}
		},
	})
}
