import AppKit
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum ImageProcessor {
    private static let supportedExtensions: Set<String> = ["jpg", "jpeg", "png", "tif", "tiff", "bmp"]

    static func imageFiles(in folder: URL) throws -> [URL] {
        let contents = try FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        return contents
            .filter { supportedExtensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }

    static func previewJPEG(from imageURL: URL) throws -> Data {
        guard let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: 1200
              ] as CFDictionary) else {
            throw CropilotError.image("Soubor \(imageURL.lastPathComponent) se nepodařilo otevřít.")
        }

        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else {
            throw CropilotError.image("Pro soubor \(imageURL.lastPathComponent) se nepodařilo vytvořit JPEG náhled.")
        }
        CGImageDestinationAddImage(destination, image, [
            kCGImageDestinationLossyCompressionQuality: 0.9
        ] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw CropilotError.image("Pro soubor \(imageURL.lastPathComponent) se nepodařilo uložit JPEG náhled.")
        }
        return data as Data
    }

    static func cropDocuments(
        images: [URL],
        scans: [Scan],
        outputFolder: URL,
        log: @escaping @Sendable (String) async -> Void
    ) async throws {
        let context = CIContext()
        for (imageURL, scan) in zip(images, scans) {
            guard let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
                  let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                await log("Soubor \(imageURL.lastPathComponent) se nepodařilo otevřít, přeskakuji.")
                continue
            }

            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
            let imageType = CGImageSourceGetType(source) ?? UTType.jpeg.identifier as CFString
            let format = outputFormat(for: imageType, fallbackExtension: imageURL.pathExtension)

            for (index, page) in scan.pages.enumerated() {
                guard let cropped = crop(image: cgImage, page: page, context: context) else {
                    await log("Stránku \(index + 1) ze souboru \(imageURL.lastPathComponent) se nepodařilo oříznout.")
                    continue
                }

                let outputURL = outputFolder
                    .appendingPathComponent(imageURL.deletingPathExtension().lastPathComponent + "_page\(index + 1)")
                    .appendingPathExtension(format.fileExtension)

                try write(cropped, to: outputURL, type: format.type, properties: properties)
                await log("Uloženo \(outputURL.lastPathComponent)")
            }
        }
    }

    private static func crop(image: CGImage, page: Page, context: CIContext) -> CGImage? {
        let width = CGFloat(image.width)
        let height = CGFloat(image.height)
        let centerX = CGFloat(page.xc) * width
        let centerYFromTop = CGFloat(page.yc) * height
        let cropWidth = CGFloat(page.width) * width
        let cropHeight = CGFloat(page.height) * height

        let ciImage = CIImage(cgImage: image)
        let center = CGPoint(x: centerX, y: height - centerYFromTop)
        let angle = CGFloat(page.angle * .pi / 180)

        let rotated = ciImage
            .transformed(by: CGAffineTransform(translationX: -center.x, y: -center.y))
            .transformed(by: CGAffineTransform(rotationAngle: angle))
            .transformed(by: CGAffineTransform(translationX: center.x, y: center.y))

        let cropRect = CGRect(
            x: center.x - cropWidth / 2,
            y: center.y - cropHeight / 2,
            width: cropWidth,
            height: cropHeight
        ).integral

        return context.createCGImage(rotated.cropped(to: cropRect), from: cropRect)
    }

    private static func write(
        _ image: CGImage,
        to url: URL,
        type: CFString,
        properties: [CFString: Any]?
    ) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, type, 1, nil) else {
            throw CropilotError.image("Soubor \(url.lastPathComponent) se nepodařilo vytvořit.")
        }
        CGImageDestinationAddImage(destination, image, properties as CFDictionary?)
        guard CGImageDestinationFinalize(destination) else {
            throw CropilotError.image("Soubor \(url.lastPathComponent) se nepodařilo uložit.")
        }
    }

    private static func outputFormat(for type: CFString, fallbackExtension: String) -> (type: CFString, fileExtension: String) {
        let typeString = type as String
        if typeString == UTType.png.identifier {
            return (UTType.png.identifier as CFString, "png")
        }
        if typeString == UTType.tiff.identifier {
            return (UTType.tiff.identifier as CFString, "tiff")
        }
        if typeString == UTType.bmp.identifier {
            return (UTType.bmp.identifier as CFString, "bmp")
        }
        let lower = fallbackExtension.lowercased()
        if lower == "jpg" || lower == "jpeg" {
            return (UTType.jpeg.identifier as CFString, lower)
        }
        return (UTType.jpeg.identifier as CFString, "jpg")
    }
}
