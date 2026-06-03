import AppKit
import Foundation

enum AppMode {
    case upload
    case download
}

enum WorkState {
    case ready
    case working
    case done
    case failed

    var label: String {
        switch self {
        case .ready:
            "Připraveno"
        case .working:
            "Pracuji"
        case .done:
            "Hotovo"
        case .failed:
            "Chyba"
        }
    }
}

@MainActor
final class CropilotViewModel: ObservableObject {
    @Published var apiKey = ""
    @Published var inputFolder = ""
    @Published var outputFolder = ""
    @Published var titleName = ""
    @Published var titleID = ""
    @Published var existingBatchInputFolder = ""
    @Published var existingBatchTitleID = ""
    @Published var downloadsExistingBatch = false
    @Published var cropModel = ""
    @Published var rotationModel = ""
    @Published var selectedMode: AppMode = .upload
    @Published var logText = ""
    @Published var state: WorkState = .ready
    @Published var showAlert = false
    @Published var alertMessage = ""
    @Published var lastEditorURL: URL?
    @Published var generatedTitleID = ""
    @Published var completionMessage = ""
    @Published var cropModels: [String] = []
    @Published var rotationModels: [String] = []
    @Published var modelStatus = "Modely se načtou po zadání API klíče."

    init() {
        if let savedKey = KeychainStore.loadAPIKey() {
            apiKey = savedKey
            modelStatus = "API klíč je uložený. Modely můžete načíst."
        }
    }

    var isWorking: Bool {
        state == .working
    }

    var canUseGuidedDownload: Bool {
        rememberedInputFolder() != nil && rememberedTitleID() != nil
    }

    func didPickInputFolder() {
        let inputURL = URL(fileURLWithPath: inputFolder)
        if outputFolder.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            outputFolder = inputURL.appendingPathComponent("cropilot_output").path
        }
    }

    func loadModels() {
        guard let key = apiKey.trimmedOrNil else {
            failValidation("Před načtením modelů zadejte skupinový API klíč.")
            return
        }
        guard !isWorking else { return }

        modelStatus = "Načítám dostupné modely..."
        Task {
            do {
                let client = try await CropilotClient(apiURL: CropilotClient.defaultAPIURL, apiKey: key) {
                    await self.append($0)
                }
                let models = try await client.loadModels()
                await MainActor.run {
                    self.cropModels = models.cropModels
                    self.rotationModels = models.rotationModels
                    self.modelStatus = "Načteno \(models.cropModels.count) modelů ořezu a \(models.rotationModels.count) modelů rotace."
                    self.rememberAPIKey(key)
                    self.trimSelectedModelsToAvailableOptions()
                }
            } catch {
                await MainActor.run {
                    self.modelStatus = "Modely se nepodařilo načíst."
                    self.append("Chyba při načítání modelů: \(error.localizedDescription)")
                    self.alertMessage = "Dostupné modely se nepodařilo načíst. Podrobnosti najdete v záznamu průběhu."
                    self.showAlert = true
                }
            }
        }
    }

    func upload() {
        guard let request = validateCommon() else { return }
        let title = titleName.trimmedOrNil ?? URL(fileURLWithPath: inputFolder).lastPathComponent
        let crop = cropModel.trimmedOrNil
        let rotation = rotationModel.trimmedOrNil
        lastEditorURL = nil
        generatedTitleID = ""
        completionMessage = ""

        run("Nahrávám obrázky...", successMessage: "Zpracování je připravené. Otevřete editor, zkontrolujte rámečky ořezu a potom stáhněte hotové výsledky.") {
            let client = try await CropilotClient(apiURL: CropilotClient.defaultAPIURL, apiKey: request.apiKey) {
                await self.append($0)
            }
            let result = try await client.uploadJob(
                inputFolder: request.inputFolder,
                cropModel: crop,
                rotationModel: rotation,
                name: title
            )
            await MainActor.run {
                self.rememberAPIKey(request.apiKey)
                self.lastEditorURL = result.editorURL
                self.generatedTitleID = result.titleID
                self.titleID = result.titleID
                self.append("ID titulu bylo vyplněno pro stažení: \(result.titleID)")
            }
        }
    }

    func download() {
        guard let key = apiKey.trimmedOrNil else {
            failValidation("Zadejte skupinový API klíč Cropilotu.")
            return
        }

        let input: URL
        let title: String
        if downloadsExistingBatch {
            guard let manualInput = validatedFolder(existingBatchInputFolder) else {
                failValidation("Vyberte původní složku se skeny pro existující dávku.")
                return
            }
            guard let manualTitle = existingBatchTitleID.trimmedOrNil else {
                failValidation("Zadejte ID existujícího titulu z Cropilotu.")
                return
            }
            input = manualInput
            title = manualTitle
        } else {
            guard let rememberedInput = rememberedInputFolder() else {
                failValidation("Nejdřív nahrajte skeny. Aplikace si potom původní složku zapamatuje pro stažení.")
                return
            }
            guard let rememberedTitle = rememberedTitleID() else {
                failValidation("Nejdřív nahrajte skeny. Aplikace si potom ID titulu zapamatuje pro stažení.")
                return
            }
            input = rememberedInput
            title = rememberedTitle
        }

        guard let output = outputFolder.trimmedOrNil else {
            failValidation("Vyberte výstupní složku.")
            return
        }

        run("Stahuji a ořezávám výsledky...", successMessage: "Oříznuté obrázky byly uloženy do vybrané výstupní složky.") {
            let client = try await CropilotClient(apiURL: CropilotClient.defaultAPIURL, apiKey: key) {
                await self.append($0)
            }
            try await client.downloadJob(
                titleID: title,
                inputFolder: input,
                outputFolder: URL(fileURLWithPath: output)
            )
            await MainActor.run {
                self.rememberAPIKey(key)
            }
        }
    }

    func openEditor() {
        if let lastEditorURL {
            NSWorkspace.shared.open(lastEditorURL)
        }
    }

    func showDownload() {
        selectedMode = .download
        downloadsExistingBatch = false
    }

    func clearLog() {
        logText = ""
    }

    func forgetAPIKey() {
        do {
            try KeychainStore.deleteAPIKey()
            apiKey = ""
            modelStatus = "API klíč byl zapomenut."
            cropModels = []
            rotationModels = []
            cropModel = ""
            rotationModel = ""
        } catch {
            append("Chyba při mazání API klíče: \(error.localizedDescription)")
            alertMessage = "API klíč se nepodařilo smazat z Keychainu."
            showAlert = true
        }
    }

    private func run(
        _ message: String,
        successMessage: String,
        operation: @escaping () async throws -> Void
    ) {
        guard !isWorking else { return }
        state = .working
        completionMessage = ""
        append(message)

        Task {
            do {
                try await operation()
                await MainActor.run {
                    self.state = .done
                    self.completionMessage = successMessage
                    self.append(successMessage)
                }
            } catch {
                await MainActor.run {
                    self.state = .failed
                    self.completionMessage = ""
                    self.append("Chyba: \(error.localizedDescription)")
                    self.alertMessage = "Operace selhala. Podrobnosti najdete v záznamu průběhu."
                    self.showAlert = true
                }
            }
        }
    }

    private func validateCommon() -> ValidatedRequest? {
        guard let key = apiKey.trimmedOrNil else {
            failValidation("Zadejte skupinový API klíč Cropilotu.")
            return nil
        }
        guard let folder = inputFolder.trimmedOrNil else {
            failValidation("Vyberte vstupní složku.")
            return nil
        }

        let folderURL = URL(fileURLWithPath: folder)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: folderURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            failValidation("Vybraná vstupní složka neexistuje.")
            return nil
        }

        let imageCount = (try? ImageProcessor.imageFiles(in: folderURL).count) ?? 0
        guard imageCount > 0 else {
            failValidation("Vybraná vstupní složka neobsahuje podporované obrázky.")
            return nil
        }

        return ValidatedRequest(apiKey: key, inputFolder: folderURL)
    }

    private func rememberedInputFolder() -> URL? {
        validatedFolder(inputFolder)
    }

    private func rememberedTitleID() -> String? {
        titleID.trimmedOrNil ?? generatedTitleID.trimmedOrNil
    }

    private func validatedFolder(_ path: String) -> URL? {
        guard let folder = path.trimmedOrNil else {
            return nil
        }
        let folderURL = URL(fileURLWithPath: folder)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: folderURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            return nil
        }
        return folderURL
    }

    private func trimSelectedModelsToAvailableOptions() {
        if !cropModel.isEmpty && !cropModels.contains(cropModel) {
            cropModel = ""
        }
        if !rotationModel.isEmpty && !rotationModels.contains(rotationModel) {
            rotationModel = ""
        }
    }

    private func failValidation(_ message: String) {
        alertMessage = message
        showAlert = true
    }

    private func rememberAPIKey(_ key: String) {
        do {
            try KeychainStore.saveAPIKey(key)
        } catch {
            append("API klíč se nepodařilo uložit: \(error.localizedDescription)")
        }
    }

    func append(_ message: String) {
        if !logText.isEmpty {
            logText += "\n"
        }
        logText += message
    }
}

private struct ValidatedRequest {
    let apiKey: String
    let inputFolder: URL
}

private extension String {
    var trimmedOrNil: String? {
        let value = trimmingCharacters(in: .whitespacesAndNewlines)
        return value.isEmpty ? nil : value
    }
}
