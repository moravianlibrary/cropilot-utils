import Foundation

typealias LogHandler = @Sendable (String) async -> Void

final class CropilotClient {
    static let defaultAPIURL = URL(string: "https://api.cropilot.trinera.cloud")!

    private let apiURL: URL
    private let apiKey: String
    private let log: LogHandler
    private let session: URLSession
    private let webURL = URL(string: "https://app.cropilot.cz")!
    private var groupID = ""
    private let statusPollIntervalSeconds: UInt64 = 10
    private let maxStatusChecks = 360

    init(apiURL: URL, apiKey: String, log: @escaping LogHandler) async throws {
        self.apiURL = apiURL
        self.apiKey = apiKey
        self.log = log
        self.session = URLSession(configuration: .default)
        try await authenticate()
    }

    func uploadJob(
        inputFolder: URL,
        cropModel: String?,
        rotationModel: String?,
        name: String
    ) async throws -> UploadResult {
        let settings = try await settings(cropModel: cropModel, rotationModel: rotationModel)
        let titleID = try await createTitle(name: name, settings: settings)
        let images = try ImageProcessor.imageFiles(in: inputFolder)

        await log("Nahrávám \(images.count) obrázků...")
        for (index, imageURL) in images.enumerated() {
            let jpegData = try ImageProcessor.previewJPEG(from: imageURL)
            let uploadName = imageURL.deletingPathExtension().lastPathComponent + ".jpg"
            try await uploadScan(titleID: titleID, filename: uploadName, data: jpegData)
            await log("[\(index + 1)/\(images.count)] Nahráno \(imageURL.lastPathComponent)")
        }

        await log("Vytvořen titul s názvem „\(name)” a ID „\(titleID)”.")
        try await process(titleID: titleID)
        let editorURL = webURL.appending(path: "book").appending(path: titleID)
        await log("Připraveno ke kontrole: \(editorURL.absoluteString)")
        return UploadResult(titleID: titleID, editorURL: editorURL)
    }

    func loadModels() async throws -> ModelOptions {
        let response: ModelsResponse = try await perform(request(path: "models", method: "GET"))
        return ModelOptions(
            cropModels: response.cropModels.sorted(),
            rotationModels: response.rotationModels.sorted()
        )
    }

    func downloadJob(titleID: String, inputFolder: URL, outputFolder: URL) async throws {
        let scans = try await downloadScans(titleID: titleID)
        try FileManager.default.createDirectory(at: outputFolder, withIntermediateDirectories: true)

        let images = try ImageProcessor.imageFiles(in: inputFolder)
        await log("Ořezávám \(min(images.count, scans.count)) obrázků...")
        try await ImageProcessor.cropDocuments(images: images, scans: scans, outputFolder: outputFolder, log: log)
        await log("Hotovo. Oříznuté obrázky byly uloženy do \(outputFolder.path)")
    }

    private func authenticate() async throws {
        let request = authorizedRequest(path: "groups", method: "GET")
        let groups: [Group] = try await perform(request)
        guard let group = groups.first else {
            throw CropilotError.api("Přihlášení proběhlo, ale API nevrátilo žádnou skupinu.")
        }
        groupID = group.id
        await log("Přihlášeno ke skupině: \(group.name)")
    }

    private func settings(cropModel: String?, rotationModel: String?) async throws -> [String: String]? {
        guard cropModel != nil || rotationModel != nil else {
            await log("Nebyl vybrán žádný model, použije se výchozí nastavení skupiny.")
            return nil
        }

        let models = try await loadModels()
        var settings: [String: String] = [:]

        if let cropModel {
            if models.cropModels.contains(cropModel) {
                settings["crop_model"] = cropModel
            } else {
                await log("Upozornění: model ořezu „\(cropModel)” nebyl nalezen, použije se výchozí model.")
            }
        }

        if let rotationModel {
            if models.rotationModels.contains(rotationModel) {
                settings["rotation_model"] = rotationModel
            } else {
                await log("Upozornění: model rotace „\(rotationModel)” nebyl nalezen, použije se výchozí model.")
            }
        }

        return settings
    }

    private func createTitle(name: String, settings: [String: String]?) async throws -> String {
        var request = authorizedRequest(path: "create", method: "POST", queryItems: [
            URLQueryItem(name: "group_id", value: groupID)
        ])
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let settingsPayload: Any = settings ?? NSNull()
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "settings": settingsPayload,
            "external_id": name
        ])

        let response: CreateResponse = try await perform(request)
        return response.id
    }

    private func uploadScan(titleID: String, filename: String, data: Data) async throws {
        let boundary = "Boundary-\(UUID().uuidString)"
        var request = authorizedRequest(path: "\(titleID)/upload-scan", method: "POST")
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        let body = MultipartFormData(boundary: boundary)
            .addingFile(field: "scan_data", filename: filename, mimeType: "image/jpeg", data: data)
            .finalized()
        try await performEmpty(request, body: body)
    }

    private func process(titleID: String) async throws {
        await log("Predikuji souřadnice ořezu stránek...")
        try await performEmpty(authorizedRequest(path: "\(titleID)/process", method: "POST"))
        await log("Cropilot zpracovává skeny. Odkaz do editoru se zobrazí, až bude zpracování hotové.")

        var lastStatus = ""
        for attempt in 1...maxStatusChecks {
            let status = try await processingStatus(titleID: titleID)
            if status.raw != lastStatus {
                await log("Stav zpracování: \(status.raw)")
                lastStatus = status.raw
            } else if attempt % 6 == 0 {
                await log("Stále čekám na Cropilot... (\(attempt * Int(statusPollIntervalSeconds)) sekund)")
            }

            switch status.state {
            case .ready:
                return
            case .failed:
                throw CropilotError.api("Zpracování v Cropilotu selhalo se stavem: \(status.raw)")
            case .waiting:
                try await Task.sleep(nanoseconds: statusPollIntervalSeconds * 1_000_000_000)
            }
        }

        throw CropilotError.api("Vypršel čas při čekání na dokončení zpracování v Cropilotu.")
    }

    private func processingStatus(titleID: String) async throws -> ProcessingStatus {
        let statusRequest = authorizedRequest(path: "\(titleID)/status", method: "GET")
        let (data, response) = try await session.data(for: statusRequest)
        try validate(response: response, data: data)
        return ProcessingStatus(data: data)
    }

    private func downloadScans(titleID: String) async throws -> [Scan] {
        let request = authorizedRequest(path: "\(titleID)/scans", method: "GET")
        let response: ScansResponse = try await perform(request)
        return response.scans
    }

    private func perform<T: Decodable>(_ request: URLRequest) async throws -> T {
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder.cropilot.decode(T.self, from: data)
    }

    private func performEmpty(_ request: URLRequest, body: Data? = nil) async throws {
        let dataAndResponse: (Data, URLResponse)
        if let body {
            dataAndResponse = try await session.upload(for: request, from: body)
        } else {
            dataAndResponse = try await session.data(for: request)
        }
        try validate(response: dataAndResponse.1, data: dataAndResponse.0)
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw CropilotError.api("Neplatná odpověď serveru.")
        }
        guard 200..<300 ~= http.statusCode else {
            let body = String(data: data, encoding: .utf8) ?? HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
            throw CropilotError.api("HTTP \(http.statusCode): \(body)")
        }
    }

    private func authorizedRequest(
        path: String,
        method: String,
        queryItems: [URLQueryItem] = []
    ) -> URLRequest {
        var request = request(path: path, method: method, queryItems: queryItems)
        request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")
        return request
    }

    private func request(
        path: String,
        method: String,
        queryItems: [URLQueryItem] = []
    ) -> URLRequest {
        let url = apiURL.appending(path: path)
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!
        if !queryItems.isEmpty {
            components.queryItems = queryItems
        }
        var request = URLRequest(url: components.url!)
        request.httpMethod = method
        return request
    }
}

enum CropilotError: LocalizedError {
    case api(String)
    case image(String)

    var errorDescription: String? {
        switch self {
        case .api(let message), .image(let message):
            message
        }
    }
}

private struct MultipartFormData {
    let boundary: String
    private var data = Data()

    init(boundary: String) {
        self.boundary = boundary
    }

    func addingFile(field: String, filename: String, mimeType: String, data fileData: Data) -> MultipartFormData {
        var copy = self
        copy.data.append("--\(boundary)\r\n")
        copy.data.append("Content-Disposition: form-data; name=\"\(field)\"; filename=\"\(filename)\"\r\n")
        copy.data.append("Content-Type: \(mimeType)\r\n\r\n")
        copy.data.append(fileData)
        copy.data.append("\r\n")
        return copy
    }

    func finalized() -> Data {
        var copy = data
        copy.append("--\(boundary)--\r\n")
        return copy
    }
}

struct Group: Decodable {
    let id: String
    let name: String

    enum CodingKeys: String, CodingKey {
        case id = "_id"
        case name
    }
}

struct ModelsResponse: Decodable {
    let cropModels: [String]
    let rotationModels: [String]

    enum CodingKeys: String, CodingKey {
        case cropModels = "crop_models"
        case rotationModels = "rotation_models"
    }
}

struct CreateResponse: Decodable {
    let id: String
}

struct ScansResponse: Decodable {
    let scans: [Scan]
}

struct Scan: Codable {
    let pages: [Page]
}

struct Page: Codable {
    let xc: Double
    let yc: Double
    let width: Double
    let height: Double
    let angle: Double
}

struct UploadResult {
    let titleID: String
    let editorURL: URL
}

struct ModelOptions {
    let cropModels: [String]
    let rotationModels: [String]
}

private struct ProcessingStatus {
    enum State {
        case waiting
        case ready
        case failed
    }

    let raw: String
    let state: State

    init(data: Data) {
        let rawText = ProcessingStatus.readableStatus(from: data)
        raw = rawText
        let normalized = rawText
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
            .lowercased()

        if ["failed", "failure", "error", "cancelled", "canceled"].contains(where: normalized.contains) {
            state = .failed
        } else if ["ready", "done", "finished", "complete", "completed", "processed", "success", "succeeded"].contains(where: normalized.contains) {
            state = .ready
        } else {
            state = .waiting
        }
    }

    private static func readableStatus(from data: Data) -> String {
        if let json = try? JSONSerialization.jsonObject(with: data) {
            return flatten(json: json).first ?? String(data: data, encoding: .utf8) ?? "unknown"
        }
        return String(data: data, encoding: .utf8) ?? "unknown"
    }

    private static func flatten(json: Any) -> [String] {
        if let string = json as? String {
            return [string]
        }
        if let number = json as? NSNumber {
            return [number.stringValue]
        }
        if let dictionary = json as? [String: Any] {
            let preferredKeys = ["status", "state", "processing_status", "processingStatus"]
            for key in preferredKeys {
                if let value = dictionary[key] {
                    let flattened = flatten(json: value)
                    if !flattened.isEmpty {
                        return flattened
                    }
                }
            }
            return dictionary.values.flatMap(flatten)
        }
        if let array = json as? [Any] {
            return array.flatMap(flatten)
        }
        return []
    }
}

private extension Data {
    mutating func append(_ string: String) {
        append(Data(string.utf8))
    }
}

extension JSONDecoder {
    static let cropilot: JSONDecoder = {
        JSONDecoder()
    }()
}

extension JSONEncoder {
    static let pretty: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }()
}
