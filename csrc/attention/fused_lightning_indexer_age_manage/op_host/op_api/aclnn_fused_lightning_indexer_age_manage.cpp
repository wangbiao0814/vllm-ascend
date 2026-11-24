#include "aclnn_fused_lightning_indexer_age_manage.h"
#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnInnerFusedLightningIndexerAgeManageGetWorkspaceSize(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, int64_t, int64_t, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
extern aclnnStatus aclnnInnerFusedLightningIndexerAgeManage(
    void *, uint64_t, aclOpExecutor *, const aclrtStream);

aclnnStatus aclnnFusedLightningIndexerAgeManageGetWorkspaceSize(
    const aclTensor *indexWeights, const aclTensor *queryDequantScale,
    const aclTensor *query, const aclTensor *indexKeyDequantScale,
    const aclTensor *indexKeyCache, const aclTensor *indexBlockTable,
    const aclTensor *actualSeqLengthsQuery, const aclTensor *actualSeqLengthsKey,
    const aclTensor *offloadSeqLengthsKey, const aclTensor *numCacheTokens,
    const aclTensor *requestState, const aclTensor *reqPoolEntries,
    const aclTensor *cacheSlotsPool, const aclTensor *sourceAgesPool, int64_t probationAge, int64_t diagnosticFlags,
    const aclTensor *topkSrcIds,
    const aclTensor *topkDstSlots, const aclTensor *topkMissCounts,
    const aclTensor *missSrcIds, const aclTensor *missDstSlots,
    const aclTensor *missCounts, const aclTensor *cacheSlotsPoolOut, const aclTensor *sourceAgesPoolOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerFusedLightningIndexerAgeManageGetWorkspaceSize(
        indexWeights, queryDequantScale, query, indexKeyDequantScale,
        indexKeyCache, indexBlockTable, actualSeqLengthsQuery,
        actualSeqLengthsKey, offloadSeqLengthsKey, numCacheTokens,
        requestState, reqPoolEntries, cacheSlotsPool, sourceAgesPool, probationAge, diagnosticFlags, topkSrcIds,
        topkDstSlots, topkMissCounts, missSrcIds, missDstSlots, missCounts,
        cacheSlotsPoolOut, sourceAgesPoolOut, workspaceSize, executor);
}
aclnnStatus aclnnFusedLightningIndexerAgeManage(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    return aclnnInnerFusedLightningIndexerAgeManage(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
